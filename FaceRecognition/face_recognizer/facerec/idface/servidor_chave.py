"""Servidor de chave do IDFace: guarda a chave secreta, decifra as pontuações
empacotadas e responde só a posição (bloco, slot) da melhor correspondência
acima do limiar τ. Nunca devolve as pontuações de todos os candidatos.

Proteções contra uso como oráculo de decifração (extensão deste projeto):
* tokens de acesso separados para identificação e administração;
* `/revelar` só decifra tokens de remoção previamente registrados, e cada
  token só pode ser revelado uma vez;
* limite de requisições por minuto e log de auditoria (sem pontuações).
"""

import hashlib
import hmac
import json
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import transformacao
from .paillier import ChaveSecreta


def hash_token(cifra: int, largura: int) -> str:
    """Hash do token (ciphertext) serializado em largura fixa."""
    return hashlib.sha256(int(cifra).to_bytes(largura, "big")).hexdigest()


class ErroServidor(Exception):
    def __init__(self, status: int, mensagem: str):
        super().__init__(mensagem)
        self.status = status


class ServidorChave:
    def __init__(self, chave: ChaveSecreta, parametros: dict, pasta_estado: Path | None = None,
                 limite_por_minuto: int = 1200):
        self.sk = chave
        self.base = int(parametros["base"])
        self.m = int(parametros["m"])
        self.beta_max = int(parametros["beta_max"])
        self.tau = {int(b): int(t) for b, t in parametros.get("tau", {}).items()}
        self.limite_por_minuto = limite_por_minuto
        self._janela = deque()
        self._trava = threading.Lock()
        self.pasta = Path(pasta_estado) if pasta_estado else None
        self._tokens = set()
        if self.pasta:
            self.pasta.mkdir(parents=True, exist_ok=True)
            arq = self.pasta / "tokens_registrados.json"
            if arq.exists():
                self._tokens = set(json.loads(arq.read_text()))

    # --- utilidades ---------------------------------------------------------
    def _auditar(self, evento: dict) -> None:
        if not self.pasta:
            return
        evento = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), **evento}
        with open(self.pasta / "auditoria.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(evento, ensure_ascii=False) + "\n")

    def _salvar_tokens(self) -> None:
        if self.pasta:
            tmp = self.pasta / "tokens_registrados.json.tmp"
            tmp.write_text(json.dumps(sorted(self._tokens)))
            tmp.replace(self.pasta / "tokens_registrados.json")

    def _limitar(self) -> None:
        agora = time.monotonic()
        with self._trava:
            while self._janela and agora - self._janela[0] > 60:
                self._janela.popleft()
            if len(self._janela) >= self.limite_por_minuto:
                raise ErroServidor(429, "limite de requisições excedido")
            self._janela.append(agora)

    # --- operações ----------------------------------------------------------
    def identificar(self, beta: int, blocos: list[dict]) -> dict:
        """``blocos``: [{"bloco": id, "slots": [ativos], "mais": hex, "menos": hex}]."""
        self._limitar()
        if beta not in self.tau:
            raise ErroServidor(400, f"sem limiar τ configurado para β={beta}")
        if beta > self.beta_max:
            raise ErroServidor(400, "β maior que o suportado pelo empacotamento")
        melhor = (None, None, -10**9)
        for b in blocos:
            mais = transformacao.decodificar(self.sk.decifrar(int(b["mais"], 16)), self.base, self.m)
            menos = transformacao.decodificar(self.sk.decifrar(int(b["menos"], 16)), self.base, self.m)
            for s in b["slots"]:
                if not 0 <= s < self.m:
                    raise ErroServidor(400, "slot inválido")
                pont = mais[s] - menos[s]
                if pont > melhor[2]:
                    melhor = (int(b["bloco"]), int(s), pont)
        encontrado = melhor[0] is not None and melhor[2] >= self.tau[beta]
        self._auditar({"op": "identificar", "blocos": len(blocos), "encontrado": encontrado})
        if not encontrado:
            return {"encontrado": False}
        return {"encontrado": True, "bloco": melhor[0], "slot": melhor[1], "pontuacao": melhor[2]}

    def registrar(self, hashes: list[str]) -> dict:
        with self._trava:
            self._tokens.update(hashes)
            self._salvar_tokens()
        self._auditar({"op": "registrar", "quantidade": len(hashes)})
        return {"ok": True}

    def revelar(self, token_hex: str) -> dict:
        h = hash_token(int(token_hex, 16), self.sk.publica.bytes_cifra)
        with self._trava:
            if h not in self._tokens:
                self._auditar({"op": "revelar", "negado": True})
                raise ErroServidor(403, "token não registrado ou já utilizado")
            self._tokens.discard(h)
            self._salvar_tokens()
        self._auditar({"op": "revelar", "token": h[:12]})
        return {"valor": format(self.sk.decifrar(int(token_hex, 16)), "x")}


def _handler(servidor: ServidorChave, token_id: str | None, token_admin: str | None):
    rotas = {
        "/identificar": ("id", lambda d: servidor.identificar(int(d["beta"]), d["blocos"])),
        "/registrar": ("admin", lambda d: servidor.registrar(list(d["hashes"]))),
        "/revelar": ("admin", lambda d: servidor.revelar(str(d["token"]))),
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # silencia o log padrão
            pass

        def _responder(self, status: int, corpo: dict):
            dados = json.dumps(corpo).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(dados)))
            self.end_headers()
            self.wfile.write(dados)

        def _autorizado(self, nivel: str) -> bool:
            esperado = token_admin if nivel == "admin" else token_id
            if esperado is None:
                return True
            recebido = self.headers.get("Authorization", "").removeprefix("Bearer ")
            return hmac.compare_digest(recebido.encode(), esperado.encode())

        def do_GET(self):
            if self.path == "/saude":
                self._responder(200, {"ok": True, "m": servidor.m, "base": servidor.base,
                                      "betas": sorted(servidor.tau)})
            else:
                self._responder(404, {"erro": "rota inexistente"})

        def do_POST(self):
            rota = rotas.get(self.path)
            if rota is None:
                return self._responder(404, {"erro": "rota inexistente"})
            if not self._autorizado(rota[0]):
                return self._responder(401, {"erro": "não autorizado"})
            try:
                tamanho = int(self.headers.get("Content-Length", "0"))
                if tamanho > 64 * 1024 * 1024:
                    raise ErroServidor(413, "requisição grande demais")
                self._responder(200, rota[1](json.loads(self.rfile.read(tamanho))))
            except ErroServidor as e:
                self._responder(e.status, {"erro": str(e)})
            except (KeyError, ValueError, TypeError) as e:
                self._responder(400, {"erro": f"requisição inválida: {e}"})

    return Handler


def servir(servidor: ServidorChave, host: str = "127.0.0.1", porta: int = 8765,
           token_id: str | None = None, token_admin: str | None = None) -> ThreadingHTTPServer:
    token_id = token_id or os.environ.get("FACEREC_IDFACE_TOKEN")
    token_admin = token_admin or os.environ.get("FACEREC_IDFACE_ADMIN")
    if host not in ("127.0.0.1", "localhost", "::1") and not (token_id and token_admin):
        raise RuntimeError("Fora de localhost, defina FACEREC_IDFACE_TOKEN e FACEREC_IDFACE_ADMIN.")
    return ThreadingHTTPServer((host, porta), _handler(servidor, token_id, token_admin))

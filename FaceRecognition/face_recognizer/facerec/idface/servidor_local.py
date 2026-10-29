"""Servidor local do IDFace: guarda a chave pública e a base cifrada (SQLite).

Layout de um bloco (até ``m`` pessoas): 2·d ciphertexts — C⁺_j e C⁻_j para
cada dimensão j — onde C^±_j = Enc(Σ_slot base^slot · x^±_{slot,j}).
"""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import transformacao as tr
from .paillier import ChavePublica, _mpz, liberar_gil
from .servidor_chave import bits_por_bloco_token, hash_token

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS meta (chave TEXT PRIMARY KEY, valor TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS blocos (id INTEGER PRIMARY KEY, cifras BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS slots (bloco INTEGER NOT NULL, slot INTEGER NOT NULL,
                                  token BLOB NOT NULL, PRIMARY KEY (bloco, slot));
"""


def parametros_publicos(pk: ChavePublica, dimensao: int, alfa: int, beta_max: int) -> dict:
    base = beta_max + 1
    return {"n": format(pk.n, "x"), "hs": format(pk.hs, "x") if pk.hs else None, "id_chave": pk.id, "dimensao": dimensao, "alfa": alfa,
            "beta_max": beta_max, "base": base, "m": tr.capacidade(pk.n, base)}


class BaseIDFace:
    def __init__(self, caminho: Path, parametros: dict, cliente, trabalhadores: int | None = None):
        hs = parametros.get("hs")
        self.pk = ChavePublica(int(parametros["n"], 16), int(hs, 16) if hs else None)
        self.d = int(parametros["dimensao"])
        self.alfa = int(parametros["alfa"])
        self.beta_max = int(parametros["beta_max"])
        self.base = int(parametros["base"])
        self.m = int(parametros["m"])
        if self.base <= self.beta_max or self.base ** self.m > self.pk.n:
            raise ValueError("parâmetros de empacotamento inválidos")
        self.cliente = cliente
        self._trab = trabalhadores
        self._cache: dict[int, list] = {}
        self._db = sqlite3.connect(Path(caminho))
        self._db.executescript(_ESQUEMA)
        atual = self._meta("id_chave")
        if atual is None:
            self._db.execute("INSERT INTO meta VALUES ('id_chave', ?)", (self.pk.id,))
            self._db.execute("INSERT INTO meta VALUES ('parametros', ?)", (json.dumps(parametros),))
            self._db.commit()
        elif atual != self.pk.id:
            raise RuntimeError("a base IDFace foi criada com outra chave pública")

    def _meta(self, chave):
        r = self._db.execute("SELECT valor FROM meta WHERE chave=?", (chave,)).fetchone()
        return r[0] if r else None

    # --- (de)serialização dos blocos ---------------------------------------
    def _ler_bloco(self, bloco: int) -> list | None:
        if bloco in self._cache:
            return self._cache[bloco]
        r = self._db.execute("SELECT cifras FROM blocos WHERE id=?", (bloco,)).fetchone()
        if r is None:
            return None
        tam = self.pk.bytes_cifra
        dados = r[0]
        cifras = [_mpz(int.from_bytes(dados[i:i + tam], "big")) for i in range(0, len(dados), tam)]
        self._cache[bloco] = cifras
        return cifras

    def _gravar_bloco(self, bloco: int, cifras: list[int]) -> None:
        dados = b"".join(self.pk.para_bytes(int(c)) for c in cifras)
        self._db.execute("INSERT OR REPLACE INTO blocos VALUES (?, ?)", (bloco, dados))
        self._cache[bloco] = [_mpz(int(c)) for c in cifras]

    def _cifrar_varios(self, valores: list[int]) -> list[int]:
        """Cifra em paralelo (com gmpy2 as exponenciações liberam o GIL)."""
        with ThreadPoolExecutor(self._trab, initializer=liberar_gil) as ex:
            return list(ex.map(self.pk.cifrar, valores, chunksize=32))

    def _cifrar_token(self, valor: int) -> bytes:
        """Token de remoção: o template (2·d bits) dividido em pedaços menores
        que n, cada um cifrado. Com chave de 2048 bits cabe num ciphertext só."""
        bits = bits_por_bloco_token(self.pk.n)
        partes = []
        while True:
            partes.append(self.pk.para_bytes(self.pk.cifrar(valor & ((1 << bits) - 1))))
            valor >>= bits
            if not valor:
                return b"".join(partes)

    def _slots_ocupados(self) -> dict[int, set[int]]:
        ocup: dict[int, set[int]] = {}
        for b, s in self._db.execute("SELECT bloco, slot FROM slots"):
            ocup.setdefault(b, set()).add(s)
        return ocup

    def _aplicar_delta(self, bloco: int, delta: np.ndarray) -> None:
        """C ← C ⊕ Enc(delta). Todas as 2·d posições recebem nova cifra (mesmo
        com delta 0), para que comparar versões da base não revele quais
        dimensões do template mudaram."""
        valores = [int(v) % self.pk.n for v in delta]
        novas = self._cifrar_varios(valores)
        atuais = self._ler_bloco(bloco)
        if atuais is not None:
            novas = [self.pk.somar(a, b) for a, b in zip(atuais, novas)]
        self._gravar_bloco(bloco, novas)

    # --- cadastro -----------------------------------------------------------
    def cadastrar(self, templates: list[np.ndarray]) -> list[tuple[int, int]]:
        """Cadastra um lote de templates (norma 1) e devolve (bloco, slot) de
        cada um. O custo é 2·d cifragens por bloco tocado, não por pessoa."""
        ocup = self._slots_ocupados()
        livres = [(b, s) for b in sorted(ocup) for s in range(self.m) if s not in ocup[b]]
        proximo = (max(ocup) + 1) if ocup else 0
        posicoes = []
        for _ in templates:
            if not livres:
                livres = [(proximo, s) for s in range(self.m)]
                proximo += 1
            posicoes.append(livres.pop(0))

        deltas: dict[int, np.ndarray] = {}
        tokens = []
        for tpl, (b, s) in zip(templates, posicoes):
            t = tr.ternaria(tpl, self.alfa)
            mais, menos = tr.separar(t)
            peso = self.base ** s
            delta = deltas.setdefault(b, np.zeros(2 * self.d, dtype=object))
            delta[: self.d] += mais.astype(object) * peso
            delta[self.d:] += menos.astype(object) * peso
            tokens.append(self._cifrar_token(tr.empacotar_token(t)))
        for b, delta in deltas.items():
            self._aplicar_delta(b, delta)
        hashes = []
        for (b, s), tok in zip(posicoes, tokens):
            self._db.execute("INSERT INTO slots VALUES (?, ?, ?)", (b, s, tok))
            hashes.append(hash_token(tok))
        self.cliente.registrar(hashes)
        self._db.commit()
        return posicoes

    def remover(self, bloco: int, slot: int) -> None:
        """Eliminação real: subtrai homomorficamente a contribuição da pessoa
        do bloco. O servidor de chave revela o token de remoção uma única vez."""
        r = self._db.execute("SELECT token FROM slots WHERE bloco=? AND slot=?", (bloco, slot)).fetchone()
        if r is None:
            raise KeyError("posição não ocupada")
        t = tr.desempacotar_token(int(self.cliente.revelar(r[0].hex()), 16), self.d)
        mais, menos = tr.separar(t)
        peso = -(self.base ** slot)
        delta = np.concatenate([mais, menos]).astype(object) * peso
        self._aplicar_delta(bloco, delta)
        self._db.execute("DELETE FROM slots WHERE bloco=? AND slot=?", (bloco, slot))
        self._db.commit()
        self._db.execute("VACUUM")

    # --- identificação --------------------------------------------------------
    def consulta_cifrada(self, template: np.ndarray, beta: int) -> list[dict]:
        """Calcula ct⁺ = IP(z⁺,C⁺) ⊕ IP(z⁻,C⁻) e ct⁻ = IP(z⁺,C⁻) ⊕ IP(z⁻,C⁺)
        para cada bloco, usando apenas multiplicações de ciphertexts."""
        if beta > self.beta_max:
            raise ValueError("β maior que β_max da base")
        z = tr.ternaria(template, beta)
        zm, zn = np.flatnonzero(z > 0), np.flatnonzero(z < 0)
        ocup = self._slots_ocupados()
        saida = []
        for b in sorted(ocup):
            c = self._ler_bloco(b)
            cmais, cmenos = c[: self.d], c[self.d:]
            ct_mais = self.pk.somar_muitos([cmais[j] for j in zm] + [cmenos[j] for j in zn])
            ct_menos = self.pk.somar_muitos([cmenos[j] for j in zm] + [cmais[j] for j in zn])
            saida.append({"bloco": b, "slots": sorted(ocup[b]),
                          "mais": format(ct_mais, "x"), "menos": format(ct_menos, "x")})
        return saida

    def identificar(self, template: np.ndarray, beta: int):
        """Devolve (bloco, slot, pontuação) ou None."""
        blocos = self.consulta_cifrada(template, beta)
        if not blocos:
            return None
        r = self.cliente.identificar(beta, blocos)
        return (r["bloco"], r["slot"], r["pontuacao"]) if r.get("encontrado") else None

    @property
    def total(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM slots").fetchone()[0]

    def fechar(self):
        self._db.close()

"""Clientes do servidor de chave: HTTP (produção) ou no mesmo processo (testes)."""

import json
import os
import urllib.error
import urllib.request


class ClienteLocal:
    """Chama um ``ServidorChave`` diretamente (sem rede)."""

    def __init__(self, servidor):
        self._s = servidor

    def identificar(self, beta, blocos):
        return self._s.identificar(beta, blocos)

    def registrar(self, hashes):
        return self._s.registrar(hashes)

    def revelar(self, token_hex):
        return self._s.revelar(token_hex)["valor"]


class ClienteHTTP:
    def __init__(self, url: str, token: str | None = None, token_admin: str | None = None,
                 tempo_limite: float = 10.0):
        self.url = url.rstrip("/")
        self.token = token or os.environ.get("FACEREC_IDFACE_TOKEN")
        self.token_admin = token_admin or os.environ.get("FACEREC_IDFACE_ADMIN")
        self.tempo_limite = tempo_limite

    def _post(self, rota: str, corpo: dict, admin: bool = False) -> dict:
        req = urllib.request.Request(self.url + rota, data=json.dumps(corpo).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        token = self.token_admin if admin else self.token
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req, timeout=self.tempo_limite) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"servidor de chave: {e.code} {e.read().decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"servidor de chave inacessível em {self.url} ({e.reason}). "
                "Inicie-o com: python idface_servidor_chave.py servir"
            ) from e

    def identificar(self, beta, blocos):
        return self._post("/identificar", {"beta": beta, "blocos": blocos})

    def registrar(self, hashes):
        return self._post("/registrar", {"hashes": hashes}, admin=True)

    def revelar(self, token_hex):
        return self._post("/revelar", {"token": token_hex}, admin=True)["valor"]

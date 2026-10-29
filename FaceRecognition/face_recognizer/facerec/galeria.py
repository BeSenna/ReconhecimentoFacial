"""Galeria de pessoas cadastradas em SQLite, com nome e template cifrados
(AES-256-GCM). Substitui o antigo ``encodings.pkl``: ``pickle.load`` executa
código arbitrário se o arquivo for adulterado, e o template ficava em claro.

Atende às boas práticas da LGPD para dado biométrico (dado pessoal sensível):
* guarda só o template (nunca a foto);
* registra quando e como o consentimento foi dado;
* prazo de retenção (``expira_em``) com expurgo automático;
* eliminação a pedido do titular (``remover``).
"""

import base64
import os
import secrets
import sqlite3
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import config

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS meta (chave TEXT PRIMARY KEY, valor TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS pessoas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT NOT NULL UNIQUE,
    nome_cifrado BLOB NOT NULL,
    template_cifrado BLOB,
    n_amostras INTEGER NOT NULL,
    criado_em TEXT NOT NULL,
    expira_em TEXT NOT NULL,
    consentimento TEXT NOT NULL,
    idface_bloco INTEGER,
    idface_slot INTEGER
);
"""


def _agora():
    return datetime.now(timezone.utc)


def carregar_ou_criar_chave(caminho: Path = config.ARQUIVO_CHAVE_GALERIA) -> bytes:
    """Chave AES de 256 bits. Prioridade: variável FACEREC_CHAVE_GALERIA
    (base64), depois arquivo com permissão 0600."""
    env = os.environ.get("FACEREC_CHAVE_GALERIA")
    if env:
        chave = base64.urlsafe_b64decode(env)
        if len(chave) != 32:
            raise ValueError("FACEREC_CHAVE_GALERIA deve ter 32 bytes em base64")
        return chave
    caminho = Path(caminho)
    if caminho.exists():
        return base64.urlsafe_b64decode(caminho.read_bytes().strip())
    caminho.parent.mkdir(parents=True, exist_ok=True)
    chave = AESGCM.generate_key(bit_length=256)
    fd = os.open(caminho, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "wb") as f:
        f.write(base64.urlsafe_b64encode(chave))
    return chave


@dataclass
class Pessoa:
    id: int
    nome: str
    template: np.ndarray | None
    n_amostras: int
    criado_em: str
    expira_em: str
    consentimento: str
    idface_bloco: int | None
    idface_slot: int | None


class Galeria:
    def __init__(self, caminho: Path = config.ARQUIVO_GALERIA, chave: bytes | None = None):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.caminho)
        self._db.executescript(_ESQUEMA)
        self._aes = AESGCM(chave or carregar_ou_criar_chave())

    # --- cifragem --------------------------------------------------------
    def _cifrar(self, dados: bytes, uid: str, campo: str) -> bytes:
        nonce = secrets.token_bytes(12)
        return nonce + self._aes.encrypt(nonce, dados, f"{uid}:{campo}".encode())

    def _decifrar(self, blob: bytes, uid: str, campo: str) -> bytes:
        return self._aes.decrypt(blob[:12], blob[12:], f"{uid}:{campo}".encode())

    # --- metadados -------------------------------------------------------
    def meta(self, chave: str, padrao=None):
        linha = self._db.execute("SELECT valor FROM meta WHERE chave=?", (chave,)).fetchone()
        return linha[0] if linha else padrao

    def definir_meta(self, chave: str, valor) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (chave, str(valor)))
        self._db.commit()

    def verificar_modelo(self, nome_modelo: str, dimensao: int, protecao: str) -> None:
        """Templates de modelos diferentes não são comparáveis; impede misturar."""
        atual = self.meta("modelo")
        if atual is None:
            self.definir_meta("modelo", nome_modelo)
            self.definir_meta("dimensao", dimensao)
            self.definir_meta("protecao", protecao)
            return
        if atual != nome_modelo or self.meta("protecao") != protecao:
            raise RuntimeError(
                f"A galeria foi criada com modelo={atual}, proteção={self.meta('protecao')}. "
                f"Para trocar para modelo={nome_modelo}, proteção={protecao}, apague "
                f"{self.caminho.name} e cadastre de novo."
            )

    # --- operações -------------------------------------------------------
    def adicionar(self, nome: str, template: np.ndarray | None, n_amostras: int,
                  consentimento: str, idface_pos: tuple[int, int] | None = None,
                  retencao_dias: int = config.RETENCAO_DIAS) -> int:
        uid = secrets.token_hex(16)
        criado = _agora()
        tpl = None
        if template is not None:
            tpl = self._cifrar(np.asarray(template, "<f4").tobytes(), uid, "template")
        bloco, slot = idface_pos if idface_pos else (None, None)
        cur = self._db.execute(
            "INSERT INTO pessoas (uid, nome_cifrado, template_cifrado, n_amostras, criado_em,"
            " expira_em, consentimento, idface_bloco, idface_slot) VALUES (?,?,?,?,?,?,?,?,?)",
            (uid, self._cifrar(nome.encode(), uid, "nome"), tpl, n_amostras,
             criado.isoformat(timespec="seconds"),
             (criado + timedelta(days=retencao_dias)).isoformat(timespec="seconds"),
             consentimento, bloco, slot),
        )
        self._db.commit()
        return int(cur.lastrowid)

    def listar(self, com_templates: bool = False) -> list[Pessoa]:
        pessoas = []
        for (id_, uid, nome_c, tpl_c, n, criado, expira, consent, bloco, slot) in self._db.execute(
            "SELECT id, uid, nome_cifrado, template_cifrado, n_amostras, criado_em, expira_em,"
            " consentimento, idface_bloco, idface_slot FROM pessoas ORDER BY id"
        ):
            tpl = None
            if com_templates and tpl_c is not None:
                tpl = np.frombuffer(self._decifrar(tpl_c, uid, "template"), "<f4").copy()
            pessoas.append(Pessoa(id_, self._decifrar(nome_c, uid, "nome").decode(), tpl, n,
                                  criado, expira, consent, bloco, slot))
        return pessoas

    def buscar_por_nome(self, nome: str) -> list[Pessoa]:
        return [p for p in self.listar() if p.nome == nome]

    def remover(self, id_: int) -> None:
        self._db.execute("DELETE FROM pessoas WHERE id=?", (id_,))
        self._db.commit()
        self._db.execute("VACUUM")  # não deixa o registro apagado no arquivo

    def expirados(self) -> list[Pessoa]:
        agora = _agora().isoformat(timespec="seconds")
        return [p for p in self.listar() if p.expira_em < agora]

    def fechar(self):
        self._db.close()

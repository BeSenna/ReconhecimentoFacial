"""Identificação 1:N com dois modos de proteção dos templates:

* ``aes``    — templates cifrados em repouso com AES-256-GCM e decifrados na
               memória para a busca (rápido; a chave fica no servidor local).
* ``idface`` — templates só existem cifrados com Paillier; a comparação é
               feita sobre os dados cifrados e só o servidor de chave, em
               outro processo/máquina, pode decifrar as pontuações.
"""

import json

import numpy as np

from . import busca, config
from .galeria import Galeria


def criar_identificador(galeria: Galeria, nome_modelo: str, protecao: str | None = None,
                        cliente_idface=None):
    protecao = protecao or galeria.meta("protecao", "aes")
    if protecao == "aes":
        return IdentificadorAES(galeria, nome_modelo)
    if protecao == "idface":
        return IdentificadorIDFace(galeria, nome_modelo, cliente_idface)
    raise ValueError(f"proteção desconhecida: {protecao}")


class IdentificadorAES:
    protecao = "aes"

    def __init__(self, galeria: Galeria, nome_modelo: str):
        self.galeria = galeria
        self.nome_modelo = nome_modelo
        self.recarregar()

    def recarregar(self):
        pessoas = [p for p in self.galeria.listar(com_templates=True) if p.template is not None]
        dim = int(self.galeria.meta("dimensao", 512))
        tpls = np.stack([p.template for p in pessoas]) if pessoas else np.zeros((0, dim), np.float32)
        self.buscador = busca.Buscador(tpls, [p.nome for p in pessoas],
                                       busca.limiar_calibrado(self.nome_modelo))

    @property
    def total(self) -> int:
        return len(self.buscador.nomes)

    def identificar(self, embeddings: np.ndarray) -> list[tuple[str | None, float]]:
        return self.buscador.buscar(embeddings)

    def cadastrar(self, itens: list[tuple[str, np.ndarray, int]], consentimento: str) -> None:
        for nome, tpl, n in itens:
            self.galeria.adicionar(nome, tpl, n, consentimento)
        self.recarregar()

    def remover(self, pessoa) -> None:
        self.galeria.remover(pessoa.id)
        self.recarregar()


def carregar_parametros_idface() -> dict:
    caminho = config.ARQUIVO_IDFACE_CHAVE_PUBLICA
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe. Gere as chaves no servidor de chave:\n"
            "  python idface_servidor_chave.py gerar-chaves\n"
            "e copie o arquivo de parâmetros públicos para esta máquina."
        )
    return json.loads(caminho.read_text())


class IdentificadorIDFace:
    protecao = "idface"

    def __init__(self, galeria: Galeria, nome_modelo: str, cliente=None, beta: int = config.IDFACE_BETA):
        from .idface.cliente import ClienteHTTP
        from .idface.servidor_local import BaseIDFace

        self.galeria = galeria
        self.nome_modelo = nome_modelo
        self.beta = beta
        self.cliente = cliente or ClienteHTTP(config.IDFACE_URL_SERVIDOR_CHAVE)
        params = carregar_parametros_idface()
        self.base = BaseIDFace(config.ARQUIVO_IDFACE_DB, params, self.cliente)
        if int(galeria.meta("dimensao", params["dimensao"])) != params["dimensao"]:
            raise RuntimeError("dimensão do modelo diferente da configurada no IDFace")
        self.fator = config.fator_idface(beta)
        self.recarregar()

    def recarregar(self):
        self._nomes = {(p.idface_bloco, p.idface_slot): p.nome for p in self.galeria.listar()
                       if p.idface_bloco is not None}

    @property
    def total(self) -> int:
        return len(self._nomes)

    def identificar(self, embeddings: np.ndarray) -> list[tuple[str | None, float]]:
        saida = []
        for e in embeddings:
            r = self.base.identificar(e, self.beta)
            if r is None:
                saida.append((None, 0.0))
            else:
                bloco, slot, pont = r
                saida.append((self._nomes.get((bloco, slot)), pont / self.fator))
        return saida

    def cadastrar(self, itens: list[tuple[str, np.ndarray, int]], consentimento: str) -> None:
        posicoes = self.base.cadastrar([tpl for _, tpl, _ in itens])
        for (nome, _, n), pos in zip(itens, posicoes):
            # nenhum template em claro é gravado: só o nome (AES) e a posição
            self.galeria.adicionar(nome, None, n, consentimento, idface_pos=pos)
        self.recarregar()

    def remover(self, pessoa) -> None:
        self.base.remover(pessoa.idface_bloco, pessoa.idface_slot)
        self.galeria.remover(pessoa.id)
        self.recarregar()

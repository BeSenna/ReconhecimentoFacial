"""Junta detecção, alinhamento, qualidade, embedding e vivacidade."""

from dataclasses import dataclass

import numpy as np

from . import deteccao, modelos, qualidade, reconhecimento


@dataclass
class Amostra:
    rosto: deteccao.Rosto
    alinhado: np.ndarray
    qualidade: qualidade.Qualidade
    embedding: np.ndarray | None = None
    prob_real: float | None = None


class Analisador:
    def __init__(self, modelo: str | None = None, vivacidade: bool = True, fiqa: bool = True):
        self.detector = deteccao.DetectorYuNet()
        self.reconhecedor = reconhecimento.criar_reconhecedor(modelo)
        self.qualidade = qualidade.AvaliadorQualidade(usar_fiqa=fiqa)
        self.vivacidade = None
        if vivacidade:
            from .vivacidade import VivacidadePassiva

            if modelos.disponivel("minifasnet"):
                self.vivacidade = VivacidadePassiva()

    @property
    def nome_modelo(self) -> str:
        return self.reconhecedor.nome

    def preparar(self, imagem_bgr, rostos, cadastro=False) -> list[Amostra]:
        amostras = []
        for r in rostos:
            try:
                alinhado = deteccao.alinhar(imagem_bgr, r.pontos)
            except ValueError:
                continue
            amostras.append(Amostra(r, alinhado, self.qualidade.avaliar(r, alinhado, cadastro)))
        return amostras

    def completar(self, imagem_bgr, amostras: list[Amostra], flip: bool,
                  com_vivacidade: bool = True) -> None:
        """Calcula embeddings (e P(real)) das amostras aprovadas, em lote."""
        boas = [a for a in amostras if a.qualidade.aprovada]
        if not boas:
            return
        embs = self.reconhecedor.embeddings([a.alinhado for a in boas], flip=flip)
        for a, e in zip(boas, embs):
            a.embedding = e
        if com_vivacidade and self.vivacidade is not None:
            probs = self.vivacidade.probabilidade_real(imagem_bgr, [a.rosto.caixa for a in boas])
            for a, p in zip(boas, probs):
                a.prob_real = float(p)

    def analisar(self, imagem_bgr, cadastro=False, flip=True, com_vivacidade=True) -> list[Amostra]:
        amostras = self.preparar(imagem_bgr, self.detector.detectar(imagem_bgr), cadastro)
        self.completar(imagem_bgr, amostras, flip, com_vivacidade)
        return amostras

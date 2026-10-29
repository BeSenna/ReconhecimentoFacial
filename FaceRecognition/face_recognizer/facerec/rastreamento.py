"""Rastreamento por IoU + votação temporal.

Cada rosto vira uma *trilha* que persiste entre frames. O reconhecimento roda
só a cada N frames por trilha e o nome exibido é decidido por maioria nos
últimos votos, o que deixa o nome estável e economiza CPU."""

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from . import config


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / uniao) if uniao > 0 else 0.0


@dataclass
class Trilha:
    id: int
    rosto: object
    perdido: int = 0
    desde_reconhecimento: int = 10**9
    votos: deque = field(default_factory=lambda: deque(maxlen=config.RASTREIO_VOTOS))
    vivacidade: deque = field(default_factory=lambda: deque(maxlen=config.RASTREIO_VOTOS))
    motivo_qualidade: str = ""
    desafio: object = None

    def votar(self, nome: str | None, similaridade: float) -> None:
        self.votos.append((nome, similaridade))

    @property
    def nome(self) -> str | None:
        """Nome com maioria absoluta entre os votos válidos (None = desconhecido
        ou ainda indeciso)."""
        if not self.votos:
            return None
        contagem = Counter(n for n, _ in self.votos)
        nome, qtd = contagem.most_common(1)[0]
        if nome is None or qtd < config.RASTREIO_VOTOS_MIN or qtd * 2 <= len(self.votos):
            return None
        return nome

    @property
    def similaridade(self) -> float:
        n = self.nome
        sims = [s for v, s in self.votos if v == n]
        return float(np.mean(sims)) if sims else 0.0

    @property
    def prob_real(self) -> float | None:
        return float(np.mean(self.vivacidade)) if self.vivacidade else None


class Rastreador:
    def __init__(self, iou_min=config.RASTREIO_IOU_MIN, max_perdido=config.RASTREIO_MAX_PERDIDO):
        self.iou_min = iou_min
        self.max_perdido = max_perdido
        self.trilhas: list[Trilha] = []
        self._proximo = 1

    def atualizar(self, rostos: list) -> list[Trilha]:
        """Associa detecções às trilhas (guloso por maior IoU) e devolve as
        trilhas visíveis neste frame."""
        pares = sorted(
            ((iou(t.rosto.caixa, r.caixa), ti, ri)
             for ti, t in enumerate(self.trilhas) for ri, r in enumerate(rostos)),
            reverse=True,
        )
        usadas_t, usadas_r = set(), set()
        for valor, ti, ri in pares:
            if valor < self.iou_min:
                break
            if ti in usadas_t or ri in usadas_r:
                continue
            t = self.trilhas[ti]
            t.rosto, t.perdido = rostos[ri], 0
            t.desde_reconhecimento += 1
            usadas_t.add(ti)
            usadas_r.add(ri)
        for ti, t in enumerate(self.trilhas):
            if ti not in usadas_t:
                t.perdido += 1
        for ri, r in enumerate(rostos):
            if ri not in usadas_r:
                self.trilhas.append(Trilha(self._proximo, r))
                self._proximo += 1
        self.trilhas = [t for t in self.trilhas if t.perdido <= self.max_perdido]
        return [t for t in self.trilhas if t.perdido == 0]

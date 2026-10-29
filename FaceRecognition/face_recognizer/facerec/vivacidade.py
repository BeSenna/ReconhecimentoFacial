"""Detecção de vivacidade (anti-spoofing).

* Passiva: MiniFASNetV2-SE (Apache-2.0) classifica o recorte como real ou ataque
  (foto impressa, tela de celular/monitor).
* Ativa: desafio aleatório de virar o rosto. Uma foto plana não muda a posição
  relativa do nariz em relação aos olhos, então não passa no desafio.
"""

import random
import time

import cv2
import numpy as np

from . import config, modelos


class VivacidadePassiva:
    TAMANHO = 128

    def __init__(self):
        import onnxruntime as ort

        o = ort.SessionOptions()
        o.log_severity_level = 3
        self._s = ort.InferenceSession(str(modelos.exigir("minifasnet")), o,
                                       providers=["CPUExecutionProvider"])
        self._entrada = self._s.get_inputs()[0].name

    @staticmethod
    def _recorte(img_rgb, caixa, expansao):
        x1, y1, x2, y2 = caixa
        lado = max(x2 - x1, y2 - y1) * expansao
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        x, y, lado = int(cx - lado / 2), int(cy - lado / 2), int(lado)
        h, w = img_rgb.shape[:2]
        recorte = img_rgb[max(0, y):min(h, y + lado), max(0, x):min(w, x + lado)]
        if recorte.size == 0:
            raise ValueError("recorte vazio")
        return cv2.copyMakeBorder(
            recorte, max(0, -y), max(0, y + lado - h), max(0, -x), max(0, x + lado - w),
            cv2.BORDER_REFLECT_101,
        )

    def _preparar(self, recorte):
        t = self.TAMANHO
        r = t / max(recorte.shape[:2])
        novo = (max(1, int(recorte.shape[1] * r)), max(1, int(recorte.shape[0] * r)))
        img = cv2.resize(recorte, novo, interpolation=cv2.INTER_LANCZOS4 if r > 1 else cv2.INTER_AREA)
        dw, dh = t - img.shape[1], t - img.shape[0]
        img = cv2.copyMakeBorder(img, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2,
                                 cv2.BORDER_REFLECT_101)
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0

    def probabilidade_real(self, imagem_bgr: np.ndarray, caixas) -> np.ndarray:
        """Retorna P(real) para cada caixa."""
        if len(caixas) == 0:
            return np.zeros(0, np.float32)
        rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
        lote = np.stack([self._preparar(self._recorte(rgb, c, config.VIVACIDADE_EXPANSAO))
                         for c in caixas])
        logits = self._s.run(None, {self._entrada: lote})[0]
        return (1.0 / (1.0 + np.exp(-(logits[:, 0] - logits[:, 1])))).astype(np.float32)


class DesafioAtivo:
    """Pede que a pessoa vire o rosto na ordem sorteada (ex.: ← depois →).

    Recebe o yaw *como a pessoa se vê na tela espelhada* (use ``-rosto.yaw``
    quando o frame processado não está espelhado), então "->" significa virar
    para a própria direita."""

    LIMIAR = 0.22
    CENTRO = 0.10

    def __init__(self, passos: int = 2, tempo_por_passo: float = 5.0, semente=None):
        rnd = random.Random(semente)
        self.sequencia = [rnd.choice([-1, 1]) for _ in range(passos)]
        if passos >= 2 and len(set(self.sequencia)) == 1:
            self.sequencia[-1] *= -1
        self.tempo_por_passo = tempo_por_passo
        self.indice = 0
        self._esperando_centro = False
        self._inicio = time.monotonic()
        self.falhou = False

    @property
    def concluido(self) -> bool:
        return self.indice >= len(self.sequencia)

    @property
    def instrucao(self) -> str:
        if self.falhou:
            return "desafio falhou"
        if self.concluido:
            return "ok"
        if self._esperando_centro:
            return "olhe para a frente"
        return "vire o rosto  " + ("<-" if self.sequencia[self.indice] < 0 else "->")

    def atualizar(self, yaw: float, agora: float | None = None) -> None:
        if self.concluido or self.falhou:
            return
        agora = time.monotonic() if agora is None else agora
        if agora - self._inicio > self.tempo_por_passo:
            self.falhou = True
            return
        if self._esperando_centro:
            if abs(yaw) < self.CENTRO:
                self._esperando_centro = False
                self._inicio = agora
            return
        direcao = self.sequencia[self.indice]
        if yaw * direcao > self.LIMIAR:
            self.indice += 1
            self._esperando_centro = True
            self._inicio = agora
        elif yaw * -direcao > self.LIMIAR:
            self.falhou = True  # virou para o lado errado (resposta gravada?)

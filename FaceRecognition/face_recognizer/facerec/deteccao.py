"""Detecção de rostos com YuNet (OpenCV, MIT) + alinhamento por 5 pontos."""

from dataclasses import dataclass

import cv2
import numpy as np

from . import config, modelos

# Posições canônicas dos 5 pontos no recorte 112x112 usado por ArcFace/SFace/AdaFace.
ARCFACE_PONTOS = np.array(
    [
        [38.2946, 51.6963],  # olho à esquerda na imagem
        [73.5318, 51.5014],  # olho à direita na imagem
        [56.0252, 71.7366],  # ponta do nariz
        [41.5493, 92.3655],  # canto da boca à esquerda
        [70.7299, 92.2041],  # canto da boca à direita
    ],
    dtype=np.float32,
)


@dataclass
class Rosto:
    caixa: np.ndarray      # x1, y1, x2, y2 (float)
    pontos: np.ndarray     # 5x2
    confianca: float

    @property
    def distancia_olhos(self) -> float:
        return float(np.linalg.norm(self.pontos[1] - self.pontos[0]))

    @property
    def yaw(self) -> float:
        """Desvio horizontal do nariz em relação ao meio dos olhos, normalizado
        pela distância entre olhos. ~0 = frontal; >0 = nariz para a direita da imagem."""
        meio = (self.pontos[0] + self.pontos[1]) / 2
        return float((self.pontos[2, 0] - meio[0]) / max(self.distancia_olhos, 1e-6))


class DetectorYuNet:
    def __init__(self, confianca=config.DETECCAO_CONFIANCA, nms=config.DETECCAO_NMS,
                 tamanho_max=config.DETECCAO_TAMANHO_MAX):
        self.tamanho_max = tamanho_max
        self._det = cv2.FaceDetectorYN.create(
            str(modelos.exigir("yunet")), "", (320, 320), confianca, nms, 5000
        )

    def detectar(self, imagem_bgr: np.ndarray) -> list[Rosto]:
        h, w = imagem_bgr.shape[:2]
        escala = min(1.0, self.tamanho_max / max(h, w))
        img = imagem_bgr
        if escala < 1.0:
            img = cv2.resize(imagem_bgr, (round(w * escala), round(h * escala)),
                             interpolation=cv2.INTER_AREA)
        self._det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self._det.detect(img)
        rostos = []
        if faces is None:
            return rostos
        for f in faces:
            x, y, bw, bh = f[:4] / escala
            pontos = (f[4:14].reshape(5, 2) / escala).astype(np.float32)
            rostos.append(Rosto(np.array([x, y, x + bw, y + bh], np.float32), pontos, float(f[14])))
        rostos.sort(key=lambda r: -(r.caixa[2] - r.caixa[0]) * (r.caixa[3] - r.caixa[1]))
        return rostos


def alinhar(imagem_bgr: np.ndarray, pontos: np.ndarray, tamanho: int = 112) -> np.ndarray:
    """Transformação de similaridade (rotação + escala + translação) que leva os
    5 pontos detectados às posições canônicas do ArcFace."""
    destino = ARCFACE_PONTOS * (tamanho / 112.0)
    matriz, _ = cv2.estimateAffinePartial2D(pontos.astype(np.float32), destino, method=cv2.LMEDS)
    if matriz is None:
        raise ValueError("não foi possível alinhar o rosto")
    return cv2.warpAffine(imagem_bgr, matriz, (tamanho, tamanho), borderValue=0)

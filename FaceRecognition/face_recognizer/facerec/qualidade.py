"""Avaliação de qualidade do rosto (FIQA): critérios geométricos + eDifFIQA."""

from dataclasses import dataclass

import cv2
import numpy as np

from . import config, modelos


@dataclass
class Qualidade:
    distancia_olhos: float
    nitidez: float
    yaw: float
    fiqa: float | None
    motivos: list[str]

    @property
    def aprovada(self) -> bool:
        return not self.motivos


class AvaliadorQualidade:
    def __init__(self, usar_fiqa: bool = True):
        self._fiqa = None
        if usar_fiqa and modelos.disponivel("ediffiqa"):
            import onnxruntime as ort

            o = ort.SessionOptions()
            o.log_severity_level = 3
            self._fiqa = ort.InferenceSession(str(modelos.caminho("ediffiqa")), o,
                                              providers=["CPUExecutionProvider"])
            self._entrada = self._fiqa.get_inputs()[0].name

    def fiqa(self, alinhado_bgr: np.ndarray) -> float | None:
        if self._fiqa is None:
            return None
        x = cv2.cvtColor(alinhado_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        x = ((x / 255.0) - 0.5) / 0.5
        x = np.ascontiguousarray(x.transpose(2, 0, 1)[None])
        return float(self._fiqa.run(None, {self._entrada: x})[0].ravel()[0])

    def avaliar(self, rosto, alinhado_bgr: np.ndarray, cadastro: bool = False) -> Qualidade:
        cinza = cv2.cvtColor(alinhado_bgr, cv2.COLOR_BGR2GRAY)
        nitidez = float(cv2.Laplacian(cinza, cv2.CV_64F).var())
        fiqa = self.fiqa(alinhado_bgr)
        motivos = []
        dist_min = config.QUALIDADE_DIST_OLHOS_MIN * (1.5 if cadastro else 1.0)
        if rosto.distancia_olhos < dist_min:
            motivos.append("rosto pequeno/distante")
        if nitidez < config.QUALIDADE_NITIDEZ_MIN:
            motivos.append("imagem borrada")
        if abs(rosto.yaw) > config.QUALIDADE_YAW_MAX * (0.7 if cadastro else 1.0):
            motivos.append("rosto muito de lado")
        limiar_fiqa = (config.QUALIDADE_FIQA_MIN_CADASTRO if cadastro
                       else config.QUALIDADE_FIQA_MIN_RECONHECIMENTO)
        if fiqa is not None and fiqa < limiar_fiqa:
            motivos.append(f"qualidade baixa (FIQA {fiqa:.2f})")
        return Qualidade(rosto.distancia_olhos, nitidez, rosto.yaw, fiqa, motivos)

"""Extração de embeddings faciais (ArcFace/SFace/AdaFace) em ONNX Runtime."""

import cv2
import numpy as np

from . import config, modelos


def normalizar(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class ReconhecedorONNX:
    """Modelo estilo ArcFace: entrada 112x112, (x - 127.5) / 127.5, NCHW.

    Com ``flip=True`` usa *flip test*: o embedding é a média do rosto e do rosto
    espelhado, técnica padrão nas avaliações (mais preciso, custa 2x)."""

    def __init__(self, caminho_modelo, nome: str, rgb: bool = True):
        import onnxruntime as ort

        self.nome = nome
        self.rgb = rgb
        opcoes = ort.SessionOptions()
        opcoes.log_severity_level = 3
        self._sessao = ort.InferenceSession(
            str(caminho_modelo), opcoes, providers=ort.get_available_providers()
        )
        self._entrada = self._sessao.get_inputs()[0].name
        self.dimensao = int(self._sessao.get_outputs()[0].shape[-1])

    def _blob(self, alinhados: list[np.ndarray], flip: bool) -> np.ndarray:
        lote = []
        for img in alinhados:
            if self.rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            lote.append(img)
            if flip:
                lote.append(img[:, ::-1])
        x = (np.stack(lote).astype(np.float32) - 127.5) / 127.5
        return np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    def embeddings(self, alinhados: list[np.ndarray], flip: bool = True) -> np.ndarray:
        if not alinhados:
            return np.zeros((0, self.dimensao), np.float32)
        saida = self._sessao.run(None, {self._entrada: self._blob(alinhados, flip)})[0]
        saida = saida.reshape(len(alinhados), 2 if flip else 1, -1).sum(axis=1)
        return normalizar(saida)


class ReconhecedorSFace:
    """SFace do OpenCV Zoo (Apache-2.0) — opção para uso comercial."""

    nome = "sface"

    def __init__(self):
        self._rec = cv2.FaceRecognizerSF.create(str(modelos.exigir("sface")), "")
        self.dimensao = 128

    def embeddings(self, alinhados: list[np.ndarray], flip: bool = True) -> np.ndarray:
        if not alinhados:
            return np.zeros((0, self.dimensao), np.float32)
        vs = []
        for img in alinhados:
            v = self._rec.feature(img).ravel()
            if flip:
                v = v + self._rec.feature(np.ascontiguousarray(img[:, ::-1])).ravel()
            vs.append(v)
        return normalizar(np.stack(vs))


def criar_reconhecedor(nome: str | None = None):
    nome = nome or config.MODELO_RECONHECIMENTO
    if nome == "arcface":
        return ReconhecedorONNX(modelos.exigir("arcface"), "arcface", rgb=True)
    if nome == "sface":
        return ReconhecedorSFace()
    if nome.startswith("onnx:"):
        # onnx:<caminho>[:bgr]  — ex.: AdaFace exportado para ONNX usa BGR
        partes = nome.split(":")
        bgr = partes[-1] == "bgr"
        caminho = ":".join(partes[1:-1] if bgr else partes[1:])
        return ReconhecedorONNX(caminho, nome, rgb=not bgr)
    raise ValueError(f"modelo de reconhecimento desconhecido: {nome}")

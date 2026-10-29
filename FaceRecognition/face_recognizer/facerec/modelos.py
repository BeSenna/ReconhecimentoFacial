"""Download e verificação (SHA-256) dos modelos ONNX."""

import hashlib
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from . import config

_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"


@dataclass(frozen=True)
class Modelo:
    arquivo: str
    url: str
    sha256: str
    licenca: str
    membro_zip: str | None = None


MODELOS = {
    "yunet": Modelo(
        "face_detection_yunet_2023mar.onnx",
        f"{_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "MIT",
    ),
    "arcface": Modelo(
        "arcface_w600k_r50.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
        "InsightFace — somente pesquisa não comercial",
        membro_zip="w600k_r50.onnx",
    ),
    "sface": Modelo(
        "face_recognition_sface_2021dec.onnx",
        f"{_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "Apache-2.0",
    ),
    "ediffiqa": Modelo(
        "ediffiqa_tiny_jun2024.onnx",
        f"{_ZOO}/face_image_quality_assessment_ediffiqa/ediffiqa_tiny_jun2024.onnx",
        "9426c899cc0f01665240cb7d9e7f98e18e24e456c178326c771a43da289bfc6a",
        "CC-BY-4.0",
    ),
    "minifasnet": Modelo(
        "minifasnet_v2se_quantized.onnx",
        "https://raw.githubusercontent.com/facenox/face-antispoof-onnx/main/models/best/98.20/best_model_quantized.onnx",
        "fde20585635cae62ed1d41796f76b6f8bc4b92cd91ec1cf0f1bc6485d2d587a9",
        "Apache-2.0",
    ),
}


def _sha256(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def caminho(nome: str) -> Path:
    return config.PASTA_MODELOS / MODELOS[nome].arquivo


def disponivel(nome: str) -> bool:
    return caminho(nome).exists()


def exigir(nome: str) -> Path:
    p = caminho(nome)
    if not p.exists():
        raise FileNotFoundError(
            f"Modelo '{nome}' não encontrado em {p}. Rode: python baixar_modelos.py"
        )
    return p


def baixar(nome: str, forcar: bool = False) -> Path:
    m = MODELOS[nome]
    destino = caminho(nome)
    if destino.exists() and not forcar:
        if _sha256(destino) == m.sha256:
            return destino
        print(f"  checksum diferente, baixando de novo: {destino.name}")
    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_arquivo = Path(tmp) / "download"
        print(f"  baixando {nome} ({m.licenca}) ...", flush=True)
        with urllib.request.urlopen(m.url) as r, open(tmp_arquivo, "wb") as f:
            shutil.copyfileobj(r, f)
        if m.membro_zip:
            with zipfile.ZipFile(tmp_arquivo) as z:
                with z.open(m.membro_zip) as src, open(Path(tmp) / "m", "wb") as dst:
                    shutil.copyfileobj(src, dst)
            tmp_arquivo = Path(tmp) / "m"
        obtido = _sha256(tmp_arquivo)
        if obtido != m.sha256:
            raise RuntimeError(f"SHA-256 inválido para {nome}: {obtido}")
        shutil.move(tmp_arquivo, destino)
    return destino


def baixar_todos(nomes=None):
    for nome in nomes or MODELOS:
        try:
            p = baixar(nome)
            print(f"✅ {nome}: {p.name}")
        except Exception as e:  # noqa: BLE001
            print(f"❌ {nome}: {e}", file=sys.stderr)

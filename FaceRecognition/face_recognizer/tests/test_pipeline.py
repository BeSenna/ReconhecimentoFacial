"""Testes que precisam dos modelos ONNX (pulados se não foram baixados)."""

import cv2
import numpy as np
import pytest

from facerec import deteccao, modelos

precisa = pytest.mark.skipif(
    not all(modelos.disponivel(m) for m in ("yunet", "arcface", "ediffiqa", "minifasnet")),
    reason="rode python baixar_modelos.py",
)


def rosto_sintetico():
    """Desenho simples de rosto (não identifica ninguém) para testes de fumaça."""
    img = np.full((480, 640, 3), 200, np.uint8)
    cv2.ellipse(img, (320, 240), (110, 145), 0, 0, 360, (140, 170, 210), -1)
    for x in (280, 360):
        cv2.ellipse(img, (x, 205), (20, 11), 0, 0, 360, (255, 255, 255), -1)
        cv2.circle(img, (x, 205), 8, (50, 30, 20), -1)
    cv2.line(img, (320, 215), (310, 270), (110, 130, 170), 4)
    cv2.ellipse(img, (320, 305), (40, 15), 0, 0, 180, (60, 60, 160), 4)
    return cv2.GaussianBlur(img, (5, 5), 0)


def test_alinhamento_leva_pontos_ao_padrao():
    # rosto "girado" 20° e em escala 2x: os 5 pontos marcados devem cair nas
    # posições canônicas do ArcFace depois do alinhamento
    ang = np.deg2rad(20)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pontos = (deteccao.ARCFACE_PONTOS - 56) @ rot.T * 2 + 200
    img = np.zeros((400, 400, 3), np.uint8)
    for x, y in pontos:
        cv2.circle(img, (int(round(x)), int(round(y))), 3, (255, 255, 255), -1)
    a = deteccao.alinhar(img, pontos.astype(np.float32))
    assert a.shape == (112, 112, 3)
    cinza = a[:, :, 0].astype(float)
    for x, y in deteccao.ARCFACE_PONTOS:
        janela = cinza[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4]
        assert janela.max() > 200


@precisa
def test_embeddings_normalizados_e_flip():
    from facerec.reconhecimento import criar_reconhecedor

    rec = criar_reconhecedor("arcface")
    img = cv2.resize(rosto_sintetico()[95:385, 210:430], (112, 112))
    e1 = rec.embeddings([img], flip=True)
    e2 = rec.embeddings([np.ascontiguousarray(img[:, ::-1])], flip=True)
    assert e1.shape == (1, 512)
    assert np.allclose(np.linalg.norm(e1, axis=1), 1, atol=1e-5)
    assert float((e1 @ e2.T).item()) > 0.999  # flip test é invariante a espelhamento


@precisa
def test_analisador_ponta_a_ponta():
    from facerec.pipeline import Analisador

    an = Analisador()
    amostras = an.analisar(rosto_sintetico(), flip=False)
    for a in amostras:
        assert a.alinhado.shape == (112, 112, 3)
        if a.qualidade.aprovada:
            assert a.embedding.shape == (512,)
            assert 0.0 <= a.prob_real <= 1.0

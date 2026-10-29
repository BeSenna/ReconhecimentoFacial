import numpy as np

from facerec.deteccao import Rosto
from facerec.rastreamento import Rastreador, iou
from facerec.vivacidade import DesafioAtivo


def rosto(x, y, lado=100):
    pontos = np.array([[x + 30, y + 40], [x + 70, y + 40], [x + 50, y + 60],
                       [x + 35, y + 80], [x + 65, y + 80]], np.float32)
    return Rosto(np.array([x, y, x + lado, y + lado], np.float32), pontos, 0.9)


def test_iou():
    a = np.array([0, 0, 10, 10.0])
    assert iou(a, a) == 1.0
    assert iou(a, np.array([20, 20, 30, 30.0])) == 0.0


def test_trilha_persiste_e_nova_pessoa_ganha_nova_trilha():
    r = Rastreador()
    t1 = r.atualizar([rosto(0, 0)])[0]
    t2 = r.atualizar([rosto(5, 3), rosto(400, 0)])
    assert t1 in t2 and len(t2) == 2
    assert r.atualizar([])  == []
    assert len(r.trilhas) == 2  # perdida por pouco tempo não é descartada


def test_votacao_estabiliza_nome():
    t = Rastreador().atualizar([rosto(0, 0)])[0]
    for nome in ["Ana", None, "Ana", "Ana", "Bruno"]:
        t.votar(nome, 0.6)
    assert t.nome == "Ana"
    t2 = Rastreador().atualizar([rosto(0, 0)])[0]
    for nome in ["Ana", "Bruno", None, "Ana"]:
        t2.votar(nome, 0.6)
    assert t2.nome is None  # sem maioria ainda


def test_desafio_ativo():
    d = DesafioAtivo(passos=2, semente=1)
    s = d.sequencia
    tempo = 0.0
    for direcao in s:
        d.atualizar(0.3 * direcao, tempo)
        d.atualizar(0.0, tempo + 0.1)
        tempo += 0.2
    assert d.concluido and not d.falhou
    errado = DesafioAtivo(passos=2, semente=1)
    errado.atualizar(-0.3 * errado.sequencia[0], 0.0)
    assert errado.falhou
    lento = DesafioAtivo(passos=1, tempo_por_passo=1.0, semente=2)
    lento.atualizar(0.0, lento._inicio + 2)
    assert lento.falhou

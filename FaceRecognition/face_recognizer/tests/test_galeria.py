import sqlite3

import numpy as np
import pytest
from cryptography.exceptions import InvalidTag

from facerec.busca import Buscador
from facerec.galeria import Galeria


@pytest.fixture
def galeria(tmp_path):
    return Galeria(tmp_path / "g.db", chave=b"k" * 32)


def test_dados_cifrados_em_repouso(galeria, tmp_path):
    tpl = np.linspace(-1, 1, 512).astype(np.float32)
    galeria.adicionar("Maria da Silva", tpl, 5, "termo assinado")
    galeria.fechar()
    bruto = (tmp_path / "g.db").read_bytes()
    assert b"Maria" not in bruto
    assert tpl.tobytes()[:64] not in bruto
    g2 = Galeria(tmp_path / "g.db", chave=b"k" * 32)
    p = g2.listar(com_templates=True)[0]
    assert p.nome == "Maria da Silva" and np.array_equal(p.template, tpl)
    with pytest.raises(InvalidTag):
        Galeria(tmp_path / "g.db", chave=b"x" * 32).listar()


def test_troca_de_cifras_entre_linhas_e_detectada(galeria, tmp_path):
    galeria.adicionar("A", np.ones(4, np.float32), 1, "c")
    galeria.adicionar("B", np.zeros(4, np.float32), 1, "c")
    db = sqlite3.connect(tmp_path / "g.db")
    (a,), (b,) = db.execute("SELECT nome_cifrado FROM pessoas ORDER BY id").fetchall()
    db.execute("UPDATE pessoas SET nome_cifrado=? WHERE id=1", (b,))
    db.commit()
    with pytest.raises(InvalidTag):
        galeria.listar()


def test_retencao_e_remocao(galeria):
    galeria.adicionar("Velho", np.ones(4, np.float32), 1, "c", retencao_dias=-1)
    galeria.adicionar("Novo", np.ones(4, np.float32), 1, "c")
    assert [p.nome for p in galeria.expirados()] == ["Velho"]
    galeria.remover(galeria.expirados()[0].id)
    assert [p.nome for p in galeria.listar()] == ["Novo"]


def test_nao_mistura_modelos(galeria):
    galeria.verificar_modelo("arcface", 512, "aes")
    galeria.verificar_modelo("arcface", 512, "aes")
    with pytest.raises(RuntimeError):
        galeria.verificar_modelo("sface", 128, "aes")


def test_buscador_limiar():
    t = np.eye(3, dtype=np.float32)
    b = Buscador(t, ["a", "b", "c"], limiar=0.5)
    r = b.buscar(np.array([[0.9, 0.1, 0], [0.3, 0.3, 0.3]], np.float32))
    assert r[0][0] == "a" and r[1][0] is None

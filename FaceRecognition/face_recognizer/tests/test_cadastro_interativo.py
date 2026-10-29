from types import SimpleNamespace

import cv2
import numpy as np

from facerec.cadastro_interativo import CadastroInterativo, montar_template
from facerec.galeria import Galeria
from facerec.identificacao import IdentificadorAES


class AnalisadorFalso:
    """Devolve sempre o mesmo rosto (com pequena variação) aprovado e vivo."""

    def __init__(self, base, vivo=True, rostos=1):
        self.base, self.vivo, self.rostos = base, vivo, rostos
        self.rng = np.random.default_rng(0)

    def analisar(self, frame, cadastro=False, flip=True, com_vivacidade=True):
        e = self.base + 0.05 * self.rng.normal(size=self.base.size)
        a = SimpleNamespace(qualidade=SimpleNamespace(aprovada=True, motivos=[]),
                            prob_real=0.99 if self.vivo else 0.1, embedding=e / np.linalg.norm(e))
        return [a] * self.rostos


def preparar(tmp_path, **kw):
    galeria = Galeria(tmp_path / "g.db", chave=b"k" * 32)
    galeria.verificar_modelo("arcface", 512, "aes")
    ident = IdentificadorAES(galeria, "arcface")
    base = np.random.default_rng(1).normal(size=512)
    an = AnalisadorFalso(base / np.linalg.norm(base), **kw)
    c = CadastroInterativo(an, ident, galeria, amostras=5, intervalo=0.0)
    tela = np.zeros((480, 640, 3), np.uint8)
    c.desenhar(tela)  # posiciona o botão
    return c, galeria, ident, tela


def clicar_no_botao(c):
    x1, y1, x2, y2 = c._botao
    c.mouse(cv2.EVENT_LBUTTONUP, (x1 + x2) // 2, (y1 + y2) // 2, 0, None)


def digitar(c, texto):
    for ch in texto:
        c.tecla(ord(ch))


def test_fluxo_completo_por_clique(tmp_path):
    c, galeria, ident, tela = preparar(tmp_path)
    c.mouse(cv2.EVENT_LBUTTONUP, 5, 5, 0, None)  # clique fora do botão
    assert not c.ativo
    clicar_no_botao(c)
    assert c.estado == c.NOME
    digitar(c, "João Silvx")
    c.tecla(8)  # backspace
    digitar(c, "a")
    c.tecla(13)
    assert c.estado == c.CONSENTIMENTO and c.nome == "João Silva"
    c.tecla(ord("S"))
    assert c.estado == c.CAPTURA
    for _ in range(5):
        c.processar(tela)
        c.desenhar(tela)
    assert c.estado == c.MENSAGEM
    pessoas = galeria.listar()
    assert [p.nome for p in pessoas] == ["João Silva"]
    assert pessoas[0].n_amostras == 5 and "confirmado na tela" in pessoas[0].consentimento
    assert ident.total == 1


def test_sem_consentimento_nada_e_gravado(tmp_path):
    c, galeria, _, _ = preparar(tmp_path)
    clicar_no_botao(c)
    digitar(c, "Ana")
    c.tecla(13)
    c.tecla(ord("n"))
    assert not c.ativo and galeria.listar() == []


def test_esc_cancela_e_q_nao_sai_durante_digitacao(tmp_path):
    c, galeria, _, _ = preparar(tmp_path)
    clicar_no_botao(c)
    assert c.tecla(ord("q")) is True  # 'q' vira letra do nome, não fecha o programa
    assert c.nome == "q"
    c.tecla(27)
    assert not c.ativo and galeria.listar() == []


def test_rejeita_foto_e_duas_pessoas(tmp_path):
    for kw, trecho in [({"vivo": False}, "foto"), ({"rostos": 2}, "uma pessoa")]:
        c, galeria, _, tela = preparar(tmp_path / trecho.replace(" ", "_"), **kw)
        clicar_no_botao(c)
        digitar(c, "X")
        c.tecla(13)
        c.tecla(ord("s"))
        for _ in range(10):
            c.processar(tela)
        assert trecho in c.status and c.embs == [] and galeria.listar() == []


def test_recadastro_substitui(tmp_path):
    c, galeria, _, tela = preparar(tmp_path)
    for _ in range(2):
        clicar_no_botao(c)
        digitar(c, "Ana")
        c.tecla(13)
        c.tecla(ord("s"))
        for _ in range(5):
            c.processar(tela)
        c.estado = c.OCIOSO
    assert [p.nome for p in galeria.listar()] == ["Ana"]


def test_montar_template_descarta_outra_pessoa():
    rng = np.random.default_rng(3)
    a = rng.normal(size=512)
    mesmas = [a + 0.1 * rng.normal(size=512) for _ in range(5)]
    outra = rng.normal(size=512)
    E = [v / np.linalg.norm(v) for v in mesmas + [outra]]
    tpl, n = montar_template(E, avisar=None)
    assert n == 5 and float(tpl @ (a / np.linalg.norm(a))) > 0.95

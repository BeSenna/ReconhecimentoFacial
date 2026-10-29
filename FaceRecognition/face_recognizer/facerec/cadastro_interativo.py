"""Cadastro pela própria janela da câmera: botão com ícone → nome →
consentimento → captura automática de amostras.

Tudo é desenhado com OpenCV (sem Tkinter/Qt extra), então funciona igual em
Linux, Windows e macOS. O teclado é lido pela janela do OpenCV.
"""

import time
import unicodedata
from datetime import datetime

import cv2
import numpy as np

from . import config
from .reconhecimento import normalizar

COERENCIA_MIN = 0.30

TECLAS_ENTER = {10, 13, 65293, 65421}
TECLAS_ESC = {27}
TECLAS_APAGAR = {8, 127, 65288}


def ascii_seguro(texto: str) -> str:
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def montar_template(embeddings, nome: str = "", avisar=print):
    """Média normalizada dos embeddings; descarta amostras que parecem ser de
    outra pessoa. Retorna (template ou None, amostras usadas)."""
    if not embeddings:
        return None, 0
    E = np.stack(embeddings)
    if len(E) >= 3:
        sims = E @ normalizar(np.median(E, axis=0))
        if np.any(sims < COERENCIA_MIN) and avisar:
            avisar(f"   ⚠️  {int(np.sum(sims < COERENCIA_MIN))} amostra(s) de {nome} parecem ser de "
                   "outra pessoa e foram descartadas")
        E = E[sims >= COERENCIA_MIN]
    return normalizar(E.mean(axis=0)), len(E)


class CadastroInterativo:
    OCIOSO, NOME, CONSENTIMENTO, CAPTURA, MENSAGEM = range(5)

    def __init__(self, analisador, identificador, galeria, amostras: int = 8,
                 intervalo: float = 0.5, tempo_limite: float = 40.0, ao_cadastrar=None):
        self.analisador = analisador
        self.ident = identificador
        self.galeria = galeria
        self.amostras = amostras
        self.intervalo = intervalo
        self.tempo_limite = tempo_limite
        self.ao_cadastrar = ao_cadastrar
        self.estado = self.OCIOSO
        self.nome = ""
        self.embs: list = []
        self.status = ""
        self._mensagem_ate = 0.0
        self._inicio = 0.0
        self._ultima = 0.0
        self._botao = (0, 0, 0, 0)
        self._sobre_botao = False

    # --- estado -------------------------------------------------------------
    @property
    def ativo(self) -> bool:
        """True enquanto o cadastro ocupa a tela (o reconhecimento pausa)."""
        return self.estado in (self.NOME, self.CONSENTIMENTO, self.CAPTURA)

    def iniciar(self):
        self.estado, self.nome, self.embs = self.NOME, "", []

    def cancelar(self, motivo="cadastro cancelado"):
        self._mostrar(motivo)

    def _mostrar(self, texto, segundos=3.0):
        self.estado = self.MENSAGEM
        self.status = texto
        self._mensagem_ate = time.monotonic() + segundos

    # --- entrada --------------------------------------------------------------
    def mouse(self, evento, x, y, *_):
        x1, y1, x2, y2 = self._botao
        dentro = x1 <= x <= x2 and y1 <= y <= y2
        self._sobre_botao = dentro
        if evento == cv2.EVENT_LBUTTONUP and dentro and not self.ativo:
            self.iniciar()

    def tecla(self, k: int) -> bool:
        """Trata a tecla; retorna True se ela foi consumida pelo cadastro."""
        if k < 0 or not self.ativo:
            return False
        if k in TECLAS_ESC:
            self.cancelar()
        elif self.estado == self.NOME:
            if k in TECLAS_ENTER:
                if self.nome.strip():
                    self.nome = self.nome.strip()
                    self.estado = self.CONSENTIMENTO
            elif k in TECLAS_APAGAR:
                self.nome = self.nome[:-1]
            else:
                c = chr(k & 0x10FFFF) if k < 0x110000 else ""
                if c.isprintable() and len(self.nome) < 40:
                    self.nome += c
        elif self.estado == self.CONSENTIMENTO:
            c = chr(k & 0xFF).lower() if k < 256 else ""
            if c == "s":
                self.estado, self.embs = self.CAPTURA, []
                self._inicio = self._ultima = time.monotonic()
                self.status = "olhe para a camera"
            elif c == "n":
                self.cancelar("sem consentimento: nada foi gravado")
        return True

    # --- processamento -------------------------------------------------------
    def processar(self, frame) -> None:
        agora = time.monotonic()
        if self.estado == self.MENSAGEM and agora > self._mensagem_ate:
            self.estado = self.OCIOSO
        if self.estado != self.CAPTURA:
            return
        if agora - self._inicio > self.tempo_limite:
            return self.cancelar("tempo esgotado - tente com mais luz e de frente")
        if agora - self._ultima < self.intervalo:
            return
        amostras = self.analisador.analisar(frame, cadastro=True, flip=True)
        if not amostras:
            self.status = "nenhum rosto - olhe para a camera"
        elif len(amostras) > 1:
            self.status = "apenas uma pessoa na frente da camera"
        elif not amostras[0].qualidade.aprovada:
            self.status = amostras[0].qualidade.motivos[0]
        elif amostras[0].prob_real is not None and amostras[0].prob_real < config.VIVACIDADE_LIMIAR:
            self.status = "parece foto/tela - use o rosto real"
        else:
            self.embs.append(amostras[0].embedding)
            self._ultima = agora
            self.status = "mova levemente o rosto"
            if len(self.embs) >= self.amostras:
                self._concluir()

    def _concluir(self):
        tpl, n = montar_template(self.embs, self.nome, avisar=None)
        if tpl is None or n < 3:
            return self.cancelar("amostras inconsistentes - tente de novo")
        for antigo in self.galeria.buscar_por_nome(self.nome):
            self.ident.remover(antigo)
        consentimento = ("confirmado na tela do reconhecer.py em "
                         + datetime.now().strftime("%d/%m/%Y %H:%M"))
        self.ident.cadastrar([(self.nome, tpl, n)], consentimento)
        print(f"✅ {self.nome} cadastrado(a) pela câmera ({n} amostras)")
        if self.ao_cadastrar:
            self.ao_cadastrar(self.nome)
        self._mostrar(f"{ascii_seguro(self.nome)} cadastrado(a)!")

    # --- desenho ----------------------------------------------------------------
    def desenhar(self, tela) -> None:
        h, w = tela.shape[:2]
        self._desenhar_botao(tela, w)
        if self.estado == self.NOME:
            self._painel(tela, ["Novo cadastro", "Digite o nome: " + ascii_seguro(self.nome) + "_",
                                "Enter confirma   Esc cancela"])
        elif self.estado == self.CONSENTIMENTO:
            self._painel(tela, [ascii_seguro(self.nome) + " concorda com o cadastro",
                                "biometrico (LGPD)?   [S] sim   [N] nao"])
        elif self.estado == self.CAPTURA:
            self._painel(tela, [f"Capturando {ascii_seguro(self.nome)}: {len(self.embs)}/{self.amostras}",
                                ascii_seguro(self.status), "Esc cancela"])
            x0, y0, larg = 20, h - 30, w - 40
            cv2.rectangle(tela, (x0, y0), (x0 + larg, y0 + 12), (80, 80, 80), cv2.FILLED)
            cv2.rectangle(tela, (x0, y0), (x0 + int(larg * len(self.embs) / self.amostras), y0 + 12),
                          (0, 200, 0), cv2.FILLED)
        elif self.estado == self.MENSAGEM:
            self._painel(tela, [ascii_seguro(self.status)])

    def _desenhar_botao(self, tela, w):
        x1, y1, x2, y2 = w - 70, 10, w - 10, 60
        self._botao = (x1, y1, x2, y2)
        cor = (0, 170, 255) if (self._sobre_botao or self.ativo) else (60, 60, 60)
        cv2.rectangle(tela, (x1, y1), (x2, y2), cor, cv2.FILLED)
        cv2.rectangle(tela, (x1, y1), (x2, y2), (255, 255, 255), 1)
        # corpo da câmera, visor e lente
        cv2.rectangle(tela, (x1 + 10, y1 + 18), (x1 + 42, y1 + 42), (255, 255, 255), 2)
        cv2.rectangle(tela, (x1 + 18, y1 + 13), (x1 + 30, y1 + 18), (255, 255, 255), cv2.FILLED)
        cv2.circle(tela, (x1 + 26, y1 + 30), 7, (255, 255, 255), 2)
        # sinal de "+"
        cv2.line(tela, (x1 + 48, y1 + 12), (x1 + 48, y1 + 24), (255, 255, 255), 2)
        cv2.line(tela, (x1 + 42, y1 + 18), (x1 + 54, y1 + 18), (255, 255, 255), 2)

    @staticmethod
    def _painel(tela, linhas):
        h, w = tela.shape[:2]
        alto = 22 + 28 * len(linhas)
        y0 = 70  # abaixo do botão, para não cobrir o rosto
        sobre = tela.copy()
        cv2.rectangle(sobre, (10, y0), (w - 80, y0 + alto), (30, 30, 30), cv2.FILLED)
        cv2.addWeighted(sobre, 0.75, tela, 0.25, 0, dst=tela)
        for i, linha in enumerate(linhas):
            cv2.putText(tela, linha, (24, y0 + 32 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2, cv2.LINE_AA)

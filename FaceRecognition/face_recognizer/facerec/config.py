"""Configuração central. Todos os caminhos são relativos à pasta do projeto,
então os scripts funcionam independentemente do diretório atual."""

import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = Path(os.environ.get("FACEREC_MODELOS", RAIZ / "models"))
PASTA_IMAGENS = RAIZ / "images"
PASTA_SAIDA = Path(os.environ.get("FACEREC_SAIDA", RAIZ / "output"))

ARQUIVO_GALERIA = PASTA_SAIDA / "galeria.db"
ARQUIVO_CHAVE_GALERIA = PASTA_SAIDA / "chave_galeria.key"
ARQUIVO_CALIBRACAO = PASTA_SAIDA / "calibracao.json"
ARQUIVO_IDFACE_DB = PASTA_SAIDA / "idface.db"
ARQUIVO_IDFACE_CHAVE_PUBLICA = PASTA_SAIDA / "idface_chave_publica.json"
# A chave secreta do IDFace pertence ao *servidor de chave*. Em produção ela
# fica em outra máquina; este caminho é apenas o padrão para testes locais.
ARQUIVO_IDFACE_CHAVE_SECRETA = PASTA_SAIDA / "servidor_chave" / "idface_chave_secreta.pem"

# --- Detecção -----------------------------------------------------------
DETECCAO_CONFIANCA = 0.7
DETECCAO_NMS = 0.3
DETECCAO_TAMANHO_MAX = 640  # o frame é reduzido para no máximo esse lado antes de detectar

# --- Reconhecimento -----------------------------------------------------
# "arcface" : ArcFace R50 (InsightFace buffalo_l, WebFace600K) — melhor precisão,
#             licença apenas para pesquisa não comercial.
# "sface"   : SFace (OpenCV Zoo) — Apache-2.0, pode ser usado comercialmente.
# "onnx:<caminho>" : qualquer modelo estilo ArcFace 112x112 (ex.: AdaFace exportado).
MODELO_RECONHECIMENTO = os.environ.get("FACEREC_MODELO", "arcface")
# Limiares de similaridade de cosseno usados se não houver calibração.
LIMIAR_PADRAO = {"arcface": 0.40, "sface": 0.36}
LIMIAR_PADRAO_GENERICO = 0.40
# "Flip test" (média do rosto e do rosto espelhado): mais preciso, custa 2x.
# Sempre usado no cadastro; no tempo real fica desligado por padrão.
FLIP_TEMPO_REAL = False

# --- Qualidade ----------------------------------------------------------
QUALIDADE_DIST_OLHOS_MIN = 24       # px no frame original
QUALIDADE_NITIDEZ_MIN = 12.0        # variância do Laplaciano (rede de segurança; o FIQA é o critério principal)
QUALIDADE_YAW_MAX = 0.35            # desvio do nariz (fração da distância entre olhos)
QUALIDADE_FIQA_MIN_CADASTRO = 0.45  # eDifFIQA (0–1, maior = melhor)
QUALIDADE_FIQA_MIN_RECONHECIMENTO = 0.25

# --- Vivacidade ---------------------------------------------------------
VIVACIDADE_LIMIAR = 0.5      # probabilidade mínima de "real"
VIVACIDADE_EXPANSAO = 1.5    # fator de expansão do recorte (igual ao treino)

# --- Rastreamento -------------------------------------------------------
RASTREIO_IOU_MIN = 0.3
RASTREIO_MAX_PERDIDO = 15        # frames sem detecção antes de descartar a trilha
RASTREIO_RECONHECER_A_CADA = 5   # reconhece cada trilha a cada N frames
RASTREIO_VOTOS = 7               # janela de votação
RASTREIO_VOTOS_MIN = 3           # votos mínimos para mostrar um nome

# --- Galeria / LGPD -----------------------------------------------------
RETENCAO_DIAS = int(os.environ.get("FACEREC_RETENCAO_DIAS", "365"))
FAISS_A_PARTIR_DE = 5000   # usa FAISS (se instalado) acima desse número de pessoas

# --- IDFace -------------------------------------------------------------
IDFACE_DIMENSAO = 512
IDFACE_ALFA = 341            # não-zeros nos templates cadastrados (artigo: 341 de 512)
IDFACE_BETA = 127            # não-zeros na consulta (artigo: 63, 127 ou 341)
IDFACE_BETA_MAX = 127        # define a base de empacotamento p = beta_max + 1
IDFACE_BITS_CHAVE = 2048     # módulo Paillier (112 bits de segurança, como no artigo)
# Relação empírica pontuação_ternária ≈ fator(β) · cosseno (ArcFace, α=341),
# medida neste projeto (r ≥ 0,98). Usada para o τ padrão e para exibir a
# similaridade; o calibrar.py mede o τ diretamente com os seus dados.
IDFACE_FATOR = {63: 80.2, 127: 146.4, 341: 288.6}
IDFACE_URL_SERVIDOR_CHAVE = os.environ.get("FACEREC_IDFACE_URL", "http://127.0.0.1:8765")


def fator_idface(beta: int) -> float:
    """Interpola linearmente IDFACE_FATOR para qualquer β."""
    pts = sorted(IDFACE_FATOR.items())
    if beta <= pts[0][0]:
        return pts[0][1] * beta / pts[0][0]
    for (b0, f0), (b1, f1) in zip(pts, pts[1:]):
        if beta <= b1:
            return f0 + (f1 - f0) * (beta - b0) / (b1 - b0)
    return pts[-1][1] * beta / pts[-1][0]

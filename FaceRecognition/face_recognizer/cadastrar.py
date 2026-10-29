"""Cadastra pessoas a partir de fotos (pasta images/) ou da webcam.

Organização das fotos (as duas formas funcionam):
    images/Maria/1.jpg, images/Maria/2.jpg, ...   (uma pasta por pessoa)
    images/Maria_1.jpg, images/Maria_2.jpg        (nome antes do primeiro "_")

Recomenda-se de 3 a 10 fotos por pessoa, de frente, bem iluminadas e com
pequenas variações de ângulo/expressão. Cada pessoa vira um único template:
a média normalizada dos embeddings das fotos aprovadas no controle de
qualidade (fotos de outra pessoa por engano são descartadas).
"""

import argparse
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from facerec import config
from facerec.camera import abrir_camera
from facerec.galeria import Galeria
from facerec.identificacao import criar_identificador
from facerec.pipeline import Analisador
from facerec.reconhecimento import normalizar

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
COERENCIA_MIN = 0.30  # amostras com similaridade menor que isso com a mediana são descartadas


def fotos_por_pessoa(pasta: Path) -> dict[str, list[Path]]:
    grupos = defaultdict(list)
    for arq in sorted(pasta.iterdir()):
        if arq.is_dir():
            grupos[arq.name].extend(sorted(f for f in arq.iterdir() if f.suffix.lower() in EXTENSOES))
        elif arq.suffix.lower() in EXTENSOES:
            grupos[arq.stem.split("_")[0]].append(arq)
    return {k: v for k, v in grupos.items() if v}


def template_de(embeddings: list[np.ndarray], nome: str) -> tuple[np.ndarray | None, int]:
    if not embeddings:
        return None, 0
    E = np.stack(embeddings)
    if len(E) >= 3:
        referencia = normalizar(np.median(E, axis=0))
        sims = E @ referencia
        if np.any(sims < COERENCIA_MIN):
            print(f"   ⚠️  {int(np.sum(sims < COERENCIA_MIN))} foto(s) de {nome} parecem ser de outra "
                  "pessoa e foram descartadas")
        E = E[sims >= COERENCIA_MIN]
    return normalizar(E.mean(axis=0)), len(E)


def embeddings_de_fotos(analisador: Analisador, fotos: list[Path]) -> list[np.ndarray]:
    embs = []
    for foto in fotos:
        img = cv2.imread(str(foto))
        if img is None:
            print(f"   ❌ {foto.name}: não foi possível abrir")
            continue
        amostras = analisador.analisar(img, cadastro=True, flip=True, com_vivacidade=False)
        if not amostras:
            print(f"   ❌ {foto.name}: nenhum rosto encontrado")
            continue
        if len(amostras) > 1:
            print(f"   ⚠️  {foto.name}: {len(amostras)} rostos, usando o maior")
        a = amostras[0]
        if not a.qualidade.aprovada:
            print(f"   ❌ {foto.name}: " + ", ".join(a.qualidade.motivos))
            continue
        fiqa = f" FIQA {a.qualidade.fiqa:.2f}" if a.qualidade.fiqa is not None else ""
        print(f"   ✅ {foto.name}{fiqa}")
        embs.append(a.embedding)
    return embs


def embeddings_da_webcam(analisador: Analisador, camera: int, quantidade: int) -> list[np.ndarray]:
    cap = abrir_camera(camera)
    if not cap.isOpened():
        sys.exit("Não foi possível abrir a webcam.")
    embs, ultimo = [], 0.0
    print("Olhe para a câmera e mova levemente o rosto. 'q' cancela.")
    while len(embs) < quantidade:
        ok, frame = cap.read()
        if not ok:
            continue
        amostras = analisador.analisar(frame, cadastro=True, flip=True)
        status = "procurando rosto"
        if len(amostras) == 1:
            a = amostras[0]
            if not a.qualidade.aprovada:
                status = ", ".join(a.qualidade.motivos)
            elif a.prob_real is not None and a.prob_real < config.VIVACIDADE_LIMIAR:
                status = "possível foto/tela — use o rosto real"
            elif time.monotonic() - ultimo > 0.6:
                embs.append(a.embedding)
                ultimo = time.monotonic()
                status = "capturado"
        elif len(amostras) > 1:
            status = "apenas uma pessoa por vez"
        tela = cv2.flip(frame, 1)
        status = unicodedata.normalize("NFKD", status).encode("ascii", "ignore").decode()
        cv2.putText(tela, f"{len(embs)}/{quantidade} - {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Cadastro", tela)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            embs = []
            break
    cap.release()
    cv2.destroyAllWindows()
    return embs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", type=Path, default=config.PASTA_IMAGENS)
    ap.add_argument("--webcam", metavar="NOME", help="cadastra NOME capturando da webcam")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--amostras", type=int, default=8, help="capturas na webcam")
    ap.add_argument("--consentimento", required=True,
                    help='como o consentimento foi obtido (LGPD art. 11), ex.: "termo assinado em 28/10/2026"')
    ap.add_argument("--protecao", choices=["aes", "idface"],
                    help="modo de proteção dos templates (padrão: o da galeria, ou aes)")
    ap.add_argument("--apagar-fotos", action="store_true",
                    help="apaga as fotos após o cadastro (minimização de dados)")
    args = ap.parse_args()

    analisador = Analisador()
    galeria = Galeria()
    protecao = args.protecao or galeria.meta("protecao", "aes")
    galeria.verificar_modelo(analisador.nome_modelo, analisador.reconhecedor.dimensao, protecao)
    ident = criar_identificador(galeria, analisador.nome_modelo, protecao)

    fontes: dict[str, list[np.ndarray]] = {}
    arquivos: dict[str, list[Path]] = {}
    if args.webcam:
        fontes[args.webcam] = embeddings_da_webcam(analisador, args.camera, args.amostras)
    else:
        if not args.pasta.exists():
            sys.exit(f"Pasta {args.pasta} não existe.")
        for nome, fotos in fotos_por_pessoa(args.pasta).items():
            print(f"\n🔍 {nome} ({len(fotos)} foto(s))")
            fontes[nome] = embeddings_de_fotos(analisador, fotos)
            arquivos[nome] = fotos

    itens = []
    for nome, embs in fontes.items():
        tpl, n = template_de(embs, nome)
        if tpl is None:
            print(f"❌ {nome}: nenhuma amostra aprovada — não cadastrado")
            continue
        if n < 3:
            print(f"⚠️  {nome}: só {n} amostra(s) boa(s); recomenda-se 3 ou mais")
        for antigo in galeria.buscar_por_nome(nome):
            ident.remover(antigo)  # recadastro substitui o anterior
        itens.append((nome, tpl, n))

    if itens:
        ident.cadastrar(itens, args.consentimento)
    print(f"\n✅ {len(itens)} pessoa(s) cadastrada(s); total na galeria: {ident.total} "
          f"(proteção: {protecao})")

    if args.apagar_fotos:
        for nome, _, _ in itens:
            for f in arquivos.get(nome, []):
                f.unlink(missing_ok=True)
        print("🗑️  fotos das pessoas cadastradas apagadas")


if __name__ == "__main__":
    main()

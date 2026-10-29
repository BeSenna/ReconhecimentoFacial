"""Reconhecimento facial em tempo real (webcam ou arquivo de vídeo).

Por trilha de rosto: qualidade → embedding → vivacidade → identificação
(AES ou IDFace) → votação temporal. Com --desafio, após reconhecer uma pessoa
viva pede que ela vire o rosto na ordem sorteada antes de liberar o acesso.
Pressione 'q' para sair.
"""

import argparse
import sys
import time
import unicodedata
from collections import Counter

import cv2

from facerec import config
from facerec.camera import abrir_camera
from facerec.galeria import Galeria
from facerec.identificacao import criar_identificador
from facerec.pipeline import Analisador
from facerec.rastreamento import Rastreador
from facerec.vivacidade import DesafioAtivo

VERDE, VERMELHO, AMARELO, AZUL = (0, 200, 0), (0, 0, 230), (0, 210, 255), (230, 160, 0)


def rotular(trilha, usar_vivacidade, usar_desafio):
    """Texto e cor do rótulo de uma trilha."""
    prob = trilha.prob_real
    if usar_vivacidade and prob is not None and prob < config.VIVACIDADE_LIMIAR \
            and len(trilha.vivacidade) >= 3:
        return f"ATAQUE? (real {prob:.0%})", VERMELHO
    if not trilha.votos:
        return (trilha.motivo_qualidade or "verificando..."), AMARELO
    nome = trilha.nome
    if nome is None:
        indeciso = sum(1 for n, _ in trilha.votos if n is not None) > 0
        return ("verificando..." if indeciso and len(trilha.votos) < config.RASTREIO_VOTOS
                else "Desconhecido"), (AMARELO if indeciso else VERMELHO)
    texto = f"{nome} ({trilha.similaridade:.0%})"
    if usar_desafio:
        d = trilha.desafio
        if d is None or not d.concluido:
            return f"{nome}: {d.instrucao if d else 'aguarde'}", AZUL
        return f"{texto} - ACESSO LIBERADO", VERDE
    return texto, VERDE


def ascii_seguro(texto: str) -> str:
    """cv2.putText não desenha acentos: "João" vira "Joao" em vez de "Jo??o"."""
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()


def espelhar_caixa(caixa, largura):
    x1, y1, x2, y2 = caixa
    return int(largura - x2), int(y1), int(largura - x1), int(y2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--video", help="processa um arquivo de vídeo em vez da webcam")
    ap.add_argument("--saida", help="grava o vídeo anotado neste arquivo (.mp4)")
    ap.add_argument("--sem-janela", action="store_true", help="não abre janela (servidor/testes)")
    ap.add_argument("--sem-vivacidade", action="store_true")
    ap.add_argument("--desafio", action="store_true", help="exige desafio ativo (virar o rosto)")
    ap.add_argument("--reconhecer-a-cada", type=int, default=config.RASTREIO_RECONHECER_A_CADA)
    args = ap.parse_args()

    galeria = Galeria()
    modelo = galeria.meta("modelo")
    if modelo is None:
        sys.exit("Galeria vazia. Cadastre pessoas primeiro: python cadastrar.py --consentimento ...")
    analisador = Analisador(modelo=modelo, vivacidade=not args.sem_vivacidade)
    ident = criar_identificador(galeria, modelo)
    usar_viv = analisador.vivacidade is not None
    if not args.sem_vivacidade and not usar_viv:
        print("⚠️  modelo de vivacidade não encontrado — rode python baixar_modelos.py")
    print(f"Galeria: {ident.total} pessoa(s), modelo {modelo}, proteção {ident.protecao}. 'q' sai.")

    cap = cv2.VideoCapture(args.video) if args.video else abrir_camera(args.camera)
    if not cap.isOpened():
        sys.exit("Não foi possível abrir a fonte de vídeo (verifique /dev/video0 e o grupo 'video').")
    gravador = None
    rastreador = Rastreador()
    t_anterior, fps = time.monotonic(), 0.0
    frames = 0
    resumo = Counter()

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if args.video:
                break
            continue
        frames += 1
        visiveis = rastreador.atualizar(analisador.detector.detectar(frame))

        pendentes = [t for t in visiveis if t.desde_reconhecimento >= args.reconhecer_a_cada]
        if pendentes:
            amostras = analisador.preparar(frame, [t.rosto for t in pendentes])
            analisador.completar(frame, amostras, flip=config.FLIP_TEMPO_REAL, com_vivacidade=usar_viv)
            por_rosto = {id(a.rosto): a for a in amostras}
            com_emb = [(t, por_rosto[id(t.rosto)]) for t in pendentes
                       if id(t.rosto) in por_rosto and por_rosto[id(t.rosto)].embedding is not None]
            try:
                resultados = ident.identificar([a.embedding for _, a in com_emb]) if com_emb else []
            except RuntimeError as e:  # servidor de chave do IDFace fora do ar
                print(f"⚠️  {e}")
                resultados = [(None, 0.0)] * len(com_emb)
            for (t, a), (nome, sim) in zip(com_emb, resultados):
                t.votar(nome, sim)
                if a.prob_real is not None:
                    t.vivacidade.append(a.prob_real)
                t.motivo_qualidade = ""
                t.desde_reconhecimento = 0
            for t in pendentes:
                a = por_rosto.get(id(t.rosto))
                if a is not None and not a.qualidade.aprovada:
                    t.motivo_qualidade = a.qualidade.motivos[0]
                    t.desde_reconhecimento = max(0, args.reconhecer_a_cada - 2)

        if args.desafio:
            for t in visiveis:
                vivo = not usar_viv or (t.prob_real or 0) >= config.VIVACIDADE_LIMIAR
                if t.nome and vivo:
                    if t.desafio is None or t.desafio.falhou:
                        t.desafio = DesafioAtivo()
                    t.desafio.atualizar(-t.rosto.yaw)  # tela espelhada

        tela = cv2.flip(frame, 1)
        largura = frame.shape[1]
        for t in visiveis:
            texto, cor = rotular(t, usar_viv, args.desafio)
            resumo[texto.split(" (")[0]] += 1
            x1, y1, x2, y2 = espelhar_caixa(t.rosto.caixa, largura)
            cv2.rectangle(tela, (x1, y1), (x2, y2), cor, 2)
            texto = ascii_seguro(texto)
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            topo = y2 if y2 + th + 12 <= tela.shape[0] else max(0, y1 - th - 12)
            x1 = min(max(0, x1), max(0, tela.shape[1] - tw - 10))
            cv2.rectangle(tela, (x1, topo), (x1 + tw + 10, topo + th + 12), cor, cv2.FILLED)
            cv2.putText(tela, texto, (x1 + 5, topo + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)

        agora = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(agora - t_anterior, 1e-6)
        t_anterior = agora
        cv2.putText(tela, f"{fps:.1f} FPS | {ident.protecao}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        if args.saida:
            if gravador is None:
                gravador = cv2.VideoWriter(args.saida, cv2.VideoWriter_fourcc(*"mp4v"),
                                           cap.get(cv2.CAP_PROP_FPS) or 25, (tela.shape[1], tela.shape[0]))
            gravador.write(tela)
        if not args.sem_janela:
            cv2.imshow("Reconhecimento Facial", tela)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if gravador is not None:
        gravador.release()
    if args.sem_janela:
        print(f"{frames} frames processados; rótulos exibidos:")
        for rotulo, qtd in resumo.most_common():
            print(f"  {qtd:5d}  {rotulo}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

"""Calibra o limiar de decisão com os SEUS dados (câmera, iluminação, pessoas).

Simula a operação real: para cada pessoa, parte das amostras forma o template
(como no cadastro) e o restante faz o papel de consultas. Mede:
* pontuações genuínas (consulta × template da própria pessoa);
* pontuações de impostor (consulta × template das outras pessoas);
e escolhe o limiar que mantém a taxa de falsa aceitação (FAR) no alvo. Faz o
mesmo para as pontuações inteiras do IDFace (τ para cada β).

Fontes (pelo menos 2 pessoas; quanto mais, melhor):
    python calibrar.py --pasta images_calibracao/        (uma pasta por pessoa)
    python calibrar.py --video Maria=maria.mp4 --video Joao=joao.mp4

Não use as mesmas fotos do cadastro em produção para estimar o limiar final:
grave vídeos curtos (10–20 s) de cada pessoa movendo levemente o rosto.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from cadastrar import fotos_por_pessoa
from facerec import config
from facerec.idface import transformacao as tr
from facerec.pipeline import Analisador
from facerec.reconhecimento import normalizar

BETAS = (63, 127, 341)


def amostras_de_imagens(analisador, fotos):
    embs = []
    for f in fotos:
        img = cv2.imread(str(f))
        if img is None:
            continue
        a = analisador.analisar(img, flip=config.FLIP_TEMPO_REAL, com_vivacidade=False)
        if a and a[0].embedding is not None:
            embs.append(a[0].embedding)
    return embs


def amostras_de_video(analisador, arquivo, passo=3):
    cap = cv2.VideoCapture(str(arquivo))
    embs, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % passo == 0:
            a = analisador.analisar(frame, flip=config.FLIP_TEMPO_REAL, com_vivacidade=False)
            if a and a[0].embedding is not None:
                embs.append(a[0].embedding)
        i += 1
    return embs


def limiar_para_far(impostor: np.ndarray, far: float) -> tuple[float, bool]:
    """Menor limiar com FAR ≤ alvo. Retorna (limiar, estimativa_confiavel)."""
    impostor = np.sort(impostor)
    confiavel = len(impostor) * far >= 10
    if confiavel:
        k = int(np.floor(len(impostor) * (1 - far)))
        return float(impostor[min(k, len(impostor) - 1)]) + 1e-6, True
    return float(impostor[-1]), False


def eer(genuino: np.ndarray, impostor: np.ndarray) -> float:
    lims = np.unique(np.concatenate([genuino, impostor]))
    frr = np.array([np.mean(genuino < t) for t in lims])
    far = np.array([np.mean(impostor >= t) for t in lims])
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2)


def pontuacoes(por_pessoa: dict[str, np.ndarray], repeticoes: int, semente: int = 0):
    rng = np.random.default_rng(semente)
    nomes = list(por_pessoa)
    cos_g, cos_i = [], []
    tern_g = {b: [] for b in BETAS}
    tern_i = {b: [] for b in BETAS}
    for _ in range(repeticoes):
        templates, consultas = {}, {}
        for n in nomes:
            E = por_pessoa[n][rng.permutation(len(por_pessoa[n]))]
            k = max(1, min(8, len(E) // 2))
            templates[n] = normalizar(E[:k].mean(axis=0))
            consultas[n] = E[k:]
        T = np.stack([templates[n] for n in nomes])
        T_tern = np.stack([tr.ternaria(t, config.IDFACE_ALFA) for t in T]).astype(np.int32)
        for i, n in enumerate(nomes):
            Q = consultas[n]
            if len(Q) == 0:
                continue
            S = Q @ T.T
            cos_g += list(S[:, i])
            cos_i += list(np.delete(S, i, axis=1).ravel())
            for b in BETAS:
                Qt = np.stack([tr.ternaria(q, b) for q in Q]).astype(np.int32)
                St = Qt @ T_tern.T
                tern_g[b] += list(St[:, i])
                tern_i[b] += list(np.delete(St, i, axis=1).ravel())
    return (np.array(cos_g), np.array(cos_i),
            {b: np.array(v) for b, v in tern_g.items()}, {b: np.array(v) for b, v in tern_i.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pasta", type=Path)
    ap.add_argument("--video", action="append", default=[], metavar="NOME=ARQUIVO")
    ap.add_argument("--far", type=float, default=1e-3, help="taxa de falsa aceitação alvo (padrão 0,1%%)")
    ap.add_argument("--repeticoes", type=int, default=10)
    ap.add_argument("--saida", type=Path, default=config.ARQUIVO_CALIBRACAO)
    args = ap.parse_args()

    analisador = Analisador(vivacidade=False)
    por_pessoa: dict[str, list] = {}
    if args.pasta:
        for nome, fotos in fotos_por_pessoa(args.pasta).items():
            por_pessoa[nome] = amostras_de_imagens(analisador, fotos)
    for item in args.video:
        nome, _, arq = item.partition("=")
        por_pessoa.setdefault(nome, []).extend(amostras_de_video(analisador, arq))
    por_pessoa = {n: np.stack(e) for n, e in por_pessoa.items() if len(e) >= 2}
    for n, e in por_pessoa.items():
        print(f"  {n}: {len(e)} amostras")
    if len(por_pessoa) < 2:
        sys.exit("São necessárias pelo menos 2 pessoas com 2+ amostras boas cada.")

    cg, ci, tg, ti = pontuacoes(por_pessoa, args.repeticoes)
    limiar, confiavel = limiar_para_far(ci, args.far)
    if not confiavel:
        padrao = config.LIMIAR_PADRAO.get(analisador.nome_modelo, config.LIMIAR_PADRAO_GENERICO)
        limiar = max(limiar + 0.05, padrao)
        print(f"\n⚠️  Só {len(ci)} pares de impostor: poucos para estimar FAR={args.far:.2%} "
              f"(precisaria de ≥{int(10 / args.far)}). Usando o maior impostor + margem, "
              f"sem ficar abaixo do padrão ({padrao}). Adicione mais pessoas.")
    print(f"\nCosseno: genuínos={len(cg)} impostores={len(ci)}  EER={eer(cg, ci):.2%}")
    print(f"  limiar={limiar:.3f}  →  TAR={np.mean(cg >= limiar):.2%}  FAR={np.mean(ci >= limiar):.3%}")

    taus = {}
    for b in BETAS:
        t, ok = limiar_para_far(ti[b].astype(float), args.far)
        if not ok:
            t = max(t + 0.05 * config.fator_idface(b), limiar * config.fator_idface(b))
        tau = int(np.ceil(t))
        taus[b] = tau
        print(f"IDFace β={b}: τ={tau}  →  TAR={np.mean(tg[b] >= tau):.2%}  "
              f"FAR={np.mean(ti[b] >= tau):.3%}  (EER {eer(tg[b], ti[b]):.2%})")

    resultado = {
        "modelo": analisador.nome_modelo,
        "limiar_cosseno": round(limiar, 4),
        "far_alvo": args.far,
        "estimativa_confiavel": confiavel,
        "tar": round(float(np.mean(cg >= limiar)), 4),
        "pares_genuinos": int(len(cg)),
        "pares_impostores": int(len(ci)),
        "idface": {"alfa": config.IDFACE_ALFA, "tau": {str(b): t for b, t in taus.items()}},
    }
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False))
    print(f"\n✅ Calibração salva em {args.saida}")
    print("   Para o IDFace, no servidor de chave: python idface_servidor_chave.py importar-calibracao")


if __name__ == "__main__":
    main()

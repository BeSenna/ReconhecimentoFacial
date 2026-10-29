"""Administra a galeria (direitos do titular na LGPD: acesso e eliminação).

    python gerenciar.py listar
    python gerenciar.py remover --nome Maria        (ou --id 3)
    python gerenciar.py expurgar                     (remove cadastros vencidos)
"""

import argparse
import sys

from facerec.galeria import Galeria
from facerec.identificacao import criar_identificador


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("listar")
    r = sub.add_parser("remover")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--id", type=int)
    g.add_argument("--nome")
    sub.add_parser("expurgar")
    args = ap.parse_args()

    galeria = Galeria()
    modelo = galeria.meta("modelo")
    protecao = galeria.meta("protecao", "aes")
    pessoas = galeria.listar()

    if args.cmd == "listar":
        print(f"Modelo: {modelo}  |  proteção: {protecao}  |  {len(pessoas)} pessoa(s)\n")
        for p in pessoas:
            print(f"[{p.id}] {p.nome}  amostras={p.n_amostras}  cadastro={p.criado_em}  "
                  f"expira={p.expira_em}  consentimento=\"{p.consentimento}\"")
        return

    alvos = (galeria.expirados() if args.cmd == "expurgar"
             else [p for p in pessoas if p.id == args.id or p.nome == args.nome])
    if not alvos:
        sys.exit("Nada a remover.")
    ident = criar_identificador(galeria, modelo, protecao)
    for p in alvos:
        ident.remover(p)
        print(f"🗑️  removido: [{p.id}] {p.nome}")


if __name__ == "__main__":
    main()

"""Baixa os modelos ONNX para ./models e confere o SHA-256 de cada um."""

import argparse
import sys

from facerec import modelos

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("modelos", nargs="*", metavar="MODELO",
                    help=f"quais baixar (padrão: todos): {', '.join(modelos.MODELOS)}")
    args = ap.parse_args()
    invalidos = set(args.modelos) - set(modelos.MODELOS)
    if invalidos:
        sys.exit(f"Modelo(s) desconhecido(s): {', '.join(sorted(invalidos))}")
    print("Licenças: ArcFace (InsightFace) é só para pesquisa não comercial; para uso "
          "comercial use FACEREC_MODELO=sface.\n")
    modelos.baixar_todos(args.modelos or None)

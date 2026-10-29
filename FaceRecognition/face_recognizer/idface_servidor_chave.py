"""Servidor de chave do IDFace (deve rodar em outra máquina/usuário em produção).

    python idface_servidor_chave.py gerar-chaves        # uma vez
    python idface_servidor_chave.py servir              # deixa rodando
    python idface_servidor_chave.py importar-calibracao # aplica τ do calibrar.py

A chave secreta é gravada cifrada com a senha da variável FACEREC_IDFACE_SENHA
(ou digitada). O arquivo de parâmetros públicos (output/idface_chave_publica.json)
é o único que o servidor local precisa receber.
"""

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from facerec import config
from facerec.idface import paillier
from facerec.idface.servidor_chave import ServidorChave, servir
from facerec.idface.servidor_local import parametros_publicos

PASTA = config.ARQUIVO_IDFACE_CHAVE_SECRETA.parent
ARQ_CONFIG = PASTA / "config.json"


def senha(confirmar=False) -> bytes | None:
    s = os.environ.get("FACEREC_IDFACE_SENHA")
    if s is None and sys.stdin.isatty():
        s = getpass.getpass("Senha da chave secreta (vazio = sem senha): ")
        if confirmar and s and getpass.getpass("Confirme: ") != s:
            sys.exit("As senhas não conferem.")
    return s.encode() if s else None


def gerar(args):
    if config.ARQUIVO_IDFACE_CHAVE_SECRETA.exists() and not args.forcar:
        sys.exit("Já existe uma chave. Use --forcar para substituir (a base cifrada antiga fica inutilizável).")
    s = senha(confirmar=True)
    if s is None:
        print("⚠️  chave secreta sem senha — proteja o arquivo e a máquina.")
    print(f"Gerando chave Paillier de {args.bits} bits...")
    sk = paillier.gerar_chaves(args.bits)
    paillier.salvar_chave_secreta(sk, config.ARQUIVO_IDFACE_CHAVE_SECRETA, s)
    params = parametros_publicos(sk.publica, config.IDFACE_DIMENSAO, args.alfa, args.beta_max)
    tau = {str(b): int(round(config.fator_idface(b) * config.LIMIAR_PADRAO_GENERICO))
           for b in (63, 127, 341) if b <= args.beta_max}
    PASTA.mkdir(parents=True, exist_ok=True)
    ARQ_CONFIG.write_text(json.dumps({**params, "tau": tau}, indent=2))
    config.ARQUIVO_IDFACE_CHAVE_PUBLICA.parent.mkdir(parents=True, exist_ok=True)
    config.ARQUIVO_IDFACE_CHAVE_PUBLICA.write_text(json.dumps(params, indent=2))
    print(f"✅ chave secreta: {config.ARQUIVO_IDFACE_CHAVE_SECRETA}")
    print(f"✅ parâmetros públicos (copie para o servidor local): {config.ARQUIVO_IDFACE_CHAVE_PUBLICA}")
    print(f"   {params['m']} templates por bloco (base {params['base']}), τ padrão {tau}")


def importar(args):
    cfg = json.loads(ARQ_CONFIG.read_text())
    cal = json.loads(Path(args.arquivo).read_text())
    taus = {b: t for b, t in cal["idface"]["tau"].items() if int(b) <= cfg["beta_max"]}
    if cal["idface"].get("alfa") != cfg["alfa"]:
        sys.exit("A calibração usou outro α.")
    cfg["tau"] = taus
    ARQ_CONFIG.write_text(json.dumps(cfg, indent=2))
    print(f"✅ τ atualizado: {taus}. Reinicie o servidor.")


def rodar(args):
    cfg = json.loads(ARQ_CONFIG.read_text())
    sk = paillier.carregar_chave_secreta(config.ARQUIVO_IDFACE_CHAVE_SECRETA, senha())
    if sk.publica.id != cfg["id_chave"]:
        sys.exit("A chave secreta não corresponde ao config.json.")
    srv = ServidorChave(sk, cfg, pasta_estado=PASTA, limite_por_minuto=args.limite)
    http = servir(srv, args.host, args.porta)
    print(f"🔐 servidor de chave em http://{args.host}:{args.porta}  (β: {sorted(srv.tau)}, τ: {srv.tau})")
    if not os.environ.get("FACEREC_IDFACE_TOKEN"):
        print("⚠️  sem FACEREC_IDFACE_TOKEN/FACEREC_IDFACE_ADMIN: aceitando só conexões locais sem autenticação")
    print("   Em rede, coloque atrás de TLS (ex.: nginx/caddy). Ctrl+C encerra.")
    try:
        http.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gerar-chaves")
    g.add_argument("--bits", type=int, default=config.IDFACE_BITS_CHAVE)
    g.add_argument("--alfa", type=int, default=config.IDFACE_ALFA)
    g.add_argument("--beta-max", type=int, default=config.IDFACE_BETA_MAX)
    g.add_argument("--forcar", action="store_true")
    s = sub.add_parser("servir")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--porta", type=int, default=8765)
    s.add_argument("--limite", type=int, default=1200, help="requisições por minuto")
    i = sub.add_parser("importar-calibracao")
    i.add_argument("arquivo", nargs="?", default=str(config.ARQUIVO_CALIBRACAO))
    args = ap.parse_args()
    {"gerar-chaves": gerar, "servir": rodar, "importar-calibracao": importar}[args.cmd](args)


if __name__ == "__main__":
    main()

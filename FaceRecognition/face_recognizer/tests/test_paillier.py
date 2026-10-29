import secrets
import sys

from facerec.idface import paillier


def test_cifra_e_decifra(chave):
    pk = chave.publica
    for m in [0, 1, 2, 12345, pk.n - 1, secrets.randbelow(pk.n)]:
        assert chave.decifrar(pk.cifrar(m)) == m


def test_cifragem_e_probabilistica(chave):
    pk = chave.publica
    assert pk.cifrar(7) != pk.cifrar(7)


def test_homomorfismo_aditivo(chave):
    pk = chave.publica
    a, b = 123456789, 987654321
    assert chave.decifrar(pk.somar(pk.cifrar(a), pk.cifrar(b))) == a + b
    assert chave.decifrar(pk.somar_muitos([pk.cifrar(i) for i in range(10)])) == 45
    assert chave.decifrar(pk.multiplicar(pk.cifrar(a), 3)) == 3 * a
    # subtração: multiplicar por −1 (mod n)
    assert chave.decifrar(pk.somar(pk.cifrar(a), pk.multiplicar(pk.cifrar(5), -1))) == a - 5


def test_sem_djn_tambem_funciona(chave):
    pk = paillier.ChavePublica(chave.publica.n)  # sem h_s → r^n clássico
    assert chave.decifrar(pk.cifrar(42)) == 42


def test_salvar_e_carregar_com_senha(chave, tmp_path):
    arq = tmp_path / "sk.pem"
    paillier.salvar_chave_secreta(chave, arq, b"senha forte")
    assert b"ENCRYPTED" in arq.read_bytes()
    sk2 = paillier.carregar_chave_secreta(arq, b"senha forte")
    assert sk2.publica.n == chave.publica.n
    assert sk2.decifrar(chave.publica.cifrar(99)) == 99
    if sys.platform != "win32":  # permissões POSIX
        assert (arq.stat().st_mode & 0o077) == 0

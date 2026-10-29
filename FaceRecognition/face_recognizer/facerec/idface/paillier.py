"""Criptossistema de Paillier (aditivamente homomórfico), com g = n + 1,
decifração por CRT e cifragem DJN (Damgård–Jurik–Nielsen): o fator aleatório
r^n é substituído por h_s^a com h_s = (−x²)^n mod n² público e a de |n|/2 bits,
a mesma otimização da biblioteca Intel IPCL usada no artigo do IDFace.
Usa gmpy2 quando instalado (bem mais rápido e libera o GIL entre threads)."""

import hashlib
import secrets
from math import gcd

try:
    import gmpy2

    def _powmod(b, e, m):
        return int(gmpy2.powmod(b, e, m))

    def _inv(a, m):
        return int(gmpy2.invert(a, m))

    def _mpz(x):
        return gmpy2.mpz(x)

    def liberar_gil():
        """Inicializador de thread: permite paralelizar exponenciações."""
        ctx = gmpy2.get_context()
        ctx.allow_release_gil = True

    GMPY2 = True
except ImportError:  # pragma: no cover - caminho sem gmpy2
    def _powmod(b, e, m):
        return pow(b, e, m)

    def _inv(a, m):
        return pow(a, -1, m)

    def _mpz(x):
        return x

    def liberar_gil():
        pass

    GMPY2 = False


class ChavePublica:
    def __init__(self, n: int, hs: int | None = None):
        self.n = int(n)
        self.n2 = self.n * self.n
        self.hs = int(hs) if hs else None
        self.bytes_cifra = (self.n2.bit_length() + 7) // 8
        self._n_mpz = _mpz(self.n)
        self._n2_mpz = _mpz(self.n2)
        self._hs_mpz = _mpz(self.hs) if self.hs else None
        self._bits_a = (self.n.bit_length() + 1) // 2

    @property
    def id(self) -> str:
        return hashlib.sha256(self.n.to_bytes((self.n.bit_length() + 7) // 8, "big")).hexdigest()[:16]

    def _ruido(self) -> int:
        if self._hs_mpz is not None:  # DJN
            return _powmod(self._hs_mpz, _mpz(secrets.randbits(self._bits_a)), self._n2_mpz)
        while True:
            r = secrets.randbelow(self.n - 1) + 1
            if gcd(r, self.n) == 1:
                return _powmod(r, self._n_mpz, self._n2_mpz)

    def cifrar(self, m: int) -> int:
        # (1 + n)^m = 1 + m·n (mod n²)
        return int((1 + (m % self.n) * self._n_mpz) * self._ruido() % self._n2_mpz)

    def somar(self, c1: int, c2: int) -> int:
        """Enc(a) ⊕ Enc(b) = Enc(a + b)."""
        return (c1 * c2) % self.n2

    def somar_muitos(self, cifras) -> int:
        acc = _mpz(1)
        n2 = self._n2_mpz
        for c in cifras:
            acc = acc * c % n2
        return int(acc)

    def multiplicar(self, c: int, k: int) -> int:
        """Enc(a) ⊗ k = Enc(k·a)."""
        return _powmod(c, k % self.n, self._n2_mpz)

    def para_bytes(self, c: int) -> bytes:
        return int(c).to_bytes(self.bytes_cifra, "big")

    def de_bytes(self, b: bytes) -> int:
        return int.from_bytes(b, "big")


class ChaveSecreta:
    def __init__(self, p: int, q: int, hs: int | None = None):
        self.p, self.q = int(p), int(q)
        self.publica = ChavePublica(self.p * self.q, hs)
        g = self.publica.n + 1
        self._p2, self._q2 = self.p * self.p, self.q * self.q
        self._hp = _inv(self._l(_powmod(g, self.p - 1, self._p2), self.p), self.p)
        self._hq = _inv(self._l(_powmod(g, self.q - 1, self._q2), self.q), self.q)
        self._q_inv_p = _inv(self.q, self.p)

    @staticmethod
    def _l(x, d):
        return (int(x) - 1) // d

    def decifrar(self, c: int) -> int:
        mp = self._l(_powmod(c, self.p - 1, self._p2), self.p) * self._hp % self.p
        mq = self._l(_powmod(c, self.q - 1, self._q2), self.q) * self._hq % self.q
        return mq + self.q * (((mp - mq) * self._q_inv_p) % self.p)


def gerar_chaves(bits: int = 2048) -> ChaveSecreta:
    """Gera n = p·q com primos de bits/2 usando o gerador RSA da biblioteca
    `cryptography` (testado e de tempo constante)."""
    from cryptography.hazmat.primitives.asymmetric import rsa

    k = rsa.generate_private_key(public_exponent=65537, key_size=bits)
    nums = k.private_numbers()
    n = nums.p * nums.q
    while True:
        x = secrets.randbelow(n - 2) + 2
        if gcd(x, n) == 1:
            break
    hs = _powmod((-x * x) % n, n, n * n)
    return ChaveSecreta(nums.p, nums.q, hs)


def salvar_chave_secreta(sk: ChaveSecreta, caminho, senha: bytes | None) -> None:
    """Salva p e q como chave RSA PKCS#8 (cifrada com a senha, se houver)."""
    import os
    import stat
    from pathlib import Path

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    p, q, n = sk.p, sk.q, sk.publica.n
    e = 65537
    d = pow(e, -1, (p - 1) * (q - 1))
    nums = rsa.RSAPrivateNumbers(p, q, d, d % (p - 1), d % (q - 1), pow(q, -1, p),
                                 rsa.RSAPublicNumbers(e, n))
    chave = nums.private_key()
    cifra = (serialization.BestAvailableEncryption(senha) if senha
             else serialization.NoEncryption())
    pem = chave.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, cifra)
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(caminho, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, stat.S_IRUSR | stat.S_IWUSR)
    with os.fdopen(fd, "wb") as f:
        f.write(pem)


def carregar_chave_secreta(caminho, senha: bytes | None) -> ChaveSecreta:
    from pathlib import Path

    from cryptography.hazmat.primitives import serialization

    chave = serialization.load_pem_private_key(Path(caminho).read_bytes(), password=senha)
    nums = chave.private_numbers()
    return ChaveSecreta(nums.p, nums.q)

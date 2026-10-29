"""Transformação quase isométrica T_k e empacotamento do IDFace."""

import numpy as np


def ternaria(template: np.ndarray, k: int) -> np.ndarray:
    """T_k: as k componentes de maior |valor| viram ±1, o resto vira 0."""
    x = np.asarray(template, dtype=np.float64).ravel()
    if not 0 < k <= x.size:
        raise ValueError(f"k deve estar em 1..{x.size}")
    t = np.zeros(x.size, dtype=np.int8)
    idx = np.argpartition(-np.abs(x), k - 1)[:k]
    t[idx] = np.where(x[idx] >= 0, 1, -1)
    return t


def separar(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """x = x⁺ − x⁻ com x⁺, x⁻ binários."""
    return (t > 0).astype(np.uint8), (t < 0).astype(np.uint8)


def pontuacao(a: np.ndarray, b: np.ndarray) -> int:
    """Pontuação em claro ⟨T_α(x), T_β(y)⟩ (referência para testes/calibração)."""
    return int(np.dot(a.astype(np.int32), b.astype(np.int32)))


def capacidade(n: int, base: int) -> int:
    """Maior m com base^m ≤ n: m dígitos < base cabem num texto claro mod n."""
    m, v = 0, 1
    while v * base <= n:
        v *= base
        m += 1
    return m


def decodificar(valor: int, base: int, m: int) -> list[int]:
    digitos = []
    for _ in range(m):
        valor, d = divmod(valor, base)
        digitos.append(d)
    return digitos


def empacotar_token(t: np.ndarray) -> int:
    """Codifica x⁺ e x⁻ num único inteiro (2·d bits) — usado no token de remoção."""
    mais, menos = separar(t)
    d = t.size
    bits = np.concatenate([mais, menos]).astype(np.uint8)
    return int.from_bytes(np.packbits(bits[::-1]).tobytes(), "big") & ((1 << (2 * d)) - 1)


def desempacotar_token(valor: int, d: int) -> np.ndarray:
    bits = np.array([(valor >> i) & 1 for i in range(2 * d)], dtype=np.int8)
    return bits[:d] - bits[d:]

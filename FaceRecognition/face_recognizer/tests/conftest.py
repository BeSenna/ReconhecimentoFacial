import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from facerec.idface import paillier  # noqa: E402


@pytest.fixture(scope="session")
def chave():
    # 1024 bits deixa os testes rápidos; em produção o padrão é 2048
    return paillier.gerar_chaves(1024)


@pytest.fixture
def templates():
    rng = np.random.default_rng(42)
    v = rng.normal(size=(40, 512))
    return v / np.linalg.norm(v, axis=1, keepdims=True)

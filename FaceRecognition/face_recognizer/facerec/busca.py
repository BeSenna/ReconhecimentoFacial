"""Busca 1:N por similaridade de cosseno (NumPy, ou FAISS para bases grandes)."""

import json

import numpy as np

from . import config


def limiar_calibrado(nome_modelo: str) -> float:
    """Limiar salvo por ``calibrar.py``; se não houver, o padrão do modelo."""
    try:
        dados = json.loads(config.ARQUIVO_CALIBRACAO.read_text())
        if dados.get("modelo") == nome_modelo:
            return float(dados["limiar_cosseno"])
    except (FileNotFoundError, KeyError, ValueError):
        pass
    return config.LIMIAR_PADRAO.get(nome_modelo, config.LIMIAR_PADRAO_GENERICO)


class Buscador:
    def __init__(self, templates: np.ndarray, nomes: list[str], limiar: float):
        self.nomes = list(nomes)
        self.limiar = limiar
        self._t = np.ascontiguousarray(templates, dtype=np.float32)
        self._faiss = None
        if len(self.nomes) >= config.FAISS_A_PARTIR_DE:
            try:
                import faiss

                self._faiss = faiss.IndexFlatIP(self._t.shape[1])
                self._faiss.add(self._t)
            except ImportError:
                pass

    def buscar(self, embeddings: np.ndarray) -> list[tuple[str | None, float]]:
        """Para cada embedding, (nome, similaridade) do melhor candidato; nome é
        None se a similaridade ficar abaixo do limiar."""
        if len(embeddings) == 0:
            return []
        if not self.nomes:
            return [(None, 0.0)] * len(embeddings)
        e = np.ascontiguousarray(embeddings, dtype=np.float32)
        if self._faiss is not None:
            sims, idx = self._faiss.search(e, 1)
            sims, idx = sims[:, 0], idx[:, 0]
        else:
            s = e @ self._t.T
            idx = s.argmax(axis=1)
            sims = s[np.arange(len(e)), idx]
        return [(self.nomes[i] if sim >= self.limiar else None, float(sim))
                for i, sim in zip(idx, sims)]

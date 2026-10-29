import json
import threading
import urllib.request

import numpy as np
import pytest

from facerec.idface import transformacao as tr
from facerec.idface.cliente import ClienteHTTP, ClienteLocal
from facerec.idface.servidor_chave import ErroServidor, ServidorChave, servir
from facerec.idface.servidor_local import BaseIDFace, parametros_publicos

ALFA, BETA = 341, 63


@pytest.fixture
def sistema(chave, tmp_path):
    params = parametros_publicos(chave.publica, 512, ALFA, BETA)
    srv = ServidorChave(chave, {**params, "tau": {BETA: 30}}, pasta_estado=tmp_path / "chave")
    base = BaseIDFace(tmp_path / "idface.db", params, ClienteLocal(srv))
    return srv, base, params


def test_ternaria():
    rng = np.random.default_rng(0)
    x = rng.normal(size=512)
    t = tr.ternaria(x, ALFA)
    assert t.dtype == np.int8 and set(np.unique(t)) <= {-1, 0, 1}
    assert np.count_nonzero(t) == ALFA
    # mantém exatamente as maiores magnitudes, com o sinal certo
    idx = np.argsort(-np.abs(x))[:ALFA]
    assert np.all(t[idx] == np.sign(x[idx]))
    mais, menos = tr.separar(t)
    assert np.all(mais.astype(int) - menos == t)


def test_token_ida_e_volta():
    t = tr.ternaria(np.random.default_rng(1).normal(size=512), ALFA)
    assert np.array_equal(tr.desempacotar_token(tr.empacotar_token(t), 512), t)


def test_capacidade_reproduz_o_artigo():
    n = 1 << 2047  # n de 2048 bits
    # artigo (Paillier 2048): 342, 293 e 243 templates para β = 63, 127, 341
    assert abs(tr.capacidade(n, 64) - 342) <= 1
    assert abs(tr.capacidade(n, 128) - 293) <= 1
    assert abs(tr.capacidade(n, 342) - 243) <= 1


def test_pontuacoes_cifradas_iguais_ao_texto_claro(sistema, chave):
    srv, base, params = sistema
    rng = np.random.default_rng(7)
    templates = rng.normal(size=(params["m"] + 15, 512))  # força mais de um bloco
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    pos = base.cadastrar(list(templates))
    assert len({b for b, _ in pos}) >= 2
    consulta = templates[7] + 0.05 * np.random.default_rng(3).normal(size=512)
    z = tr.ternaria(consulta, BETA)
    for b in base.consulta_cifrada(consulta, BETA):
        mais = tr.decodificar(chave.decifrar(int(b["mais"], 16)), params["base"], params["m"])
        menos = tr.decodificar(chave.decifrar(int(b["menos"], 16)), params["base"], params["m"])
        for s in b["slots"]:
            i = pos.index((b["bloco"], s))
            assert mais[s] - menos[s] == tr.pontuacao(tr.ternaria(templates[i], ALFA), z)


def test_identifica_e_rejeita(sistema, templates):
    _, base, _ = sistema
    pos = base.cadastrar(list(templates))
    r = base.identificar(templates[12] + 0.02 * np.random.default_rng(5).normal(size=512), BETA)
    assert r is not None and (r[0], r[1]) == pos[12]
    estranho = np.random.default_rng(99).normal(size=512)
    assert base.identificar(estranho / np.linalg.norm(estranho), BETA) is None


def test_remocao_real_e_reuso_do_slot(sistema, templates):
    srv, base, _ = sistema
    pos = base.cadastrar(list(templates[:5]))
    base.remover(*pos[2])
    assert base.total == 4
    assert base.identificar(templates[2], BETA) is None
    # o bloco volta a ter exatamente as contribuições das outras pessoas
    assert (base.identificar(templates[3], BETA) or (None, None))[:2] == pos[3]
    with pytest.raises(KeyError):
        base.remover(*pos[2])
    novo = base.cadastrar([templates[30]])
    assert novo == [pos[2]]
    assert base.identificar(templates[30], BETA)[:2] == pos[2]


def test_banco_nao_muda_so_onde_o_template_tem_bits(sistema, templates):
    """Toda atualização recifra todas as posições: comparar duas versões do
    banco não revela o suporte do template cadastrado."""
    _, base, _ = sistema
    base.cadastrar([templates[0]])
    antes = list(base._ler_bloco(0))
    base.cadastrar([templates[1]])
    depois = base._ler_bloco(0)
    assert all(a != d for a, d in zip(antes, depois))


def test_servidor_chave_nao_e_oraculo(sistema, chave, templates):
    srv, base, params = sistema
    base.cadastrar(list(templates[:3]))
    bloco = base._ler_bloco(0)
    with pytest.raises(ErroServidor) as e:  # cifra do banco não é um token registrado
        srv.revelar(format(int(bloco[0]), "x"))
    assert e.value.status == 403
    token = base._db.execute("SELECT token FROM slots LIMIT 1").fetchone()[0].hex()
    srv.revelar(token)
    with pytest.raises(ErroServidor):  # uso único
        srv.revelar(token)
    with pytest.raises(ErroServidor):  # β sem τ configurado
        srv.identificar(127, [])
    log = (srv.pasta / "auditoria.log").read_text()
    assert "pontuacao" not in log


def test_limite_de_requisicoes(chave):
    params = parametros_publicos(chave.publica, 512, ALFA, BETA)
    srv = ServidorChave(chave, {**params, "tau": {BETA: 1}}, limite_por_minuto=3)
    for _ in range(3):
        srv.identificar(BETA, [])
    with pytest.raises(ErroServidor) as e:
        srv.identificar(BETA, [])
    assert e.value.status == 429


def test_http_com_autenticacao(chave, tmp_path, templates):
    params = parametros_publicos(chave.publica, 512, ALFA, BETA)
    srv = ServidorChave(chave, {**params, "tau": {BETA: 30}}, pasta_estado=tmp_path)
    http = servir(srv, "127.0.0.1", 0, token_id="abc", token_admin="adm")
    threading.Thread(target=http.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{http.server_address[1]}"
    try:
        base = BaseIDFace(tmp_path / "db", params, ClienteHTTP(url, "abc", "adm"))
        pos = base.cadastrar(list(templates[:4]))
        assert base.identificar(templates[1], BETA)[:2] == pos[1]
        base.remover(*pos[0])
        sem_token = ClienteHTTP(url, "errado", "errado")
        with pytest.raises(RuntimeError, match="401"):
            sem_token.identificar(BETA, [])
        with urllib.request.urlopen(url + "/saude") as r:
            assert json.loads(r.read())["m"] == params["m"]
    finally:
        http.shutdown()

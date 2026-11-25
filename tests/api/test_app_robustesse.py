"""Tâche A1 : cas limites du middleware, de /health et du contrat.

Tests issus de l'injection d'erreurs de la boucle 2 de vérification. Chacun tue au moins
une mutation (identifiant entre crochets). Le middleware est souvent appelé en ASGI direct :
TestClient livre le corps en un seul message et ne permet pas de tester la lecture par morceaux.
"""

import asyncio
import json
import logging
import threading
import time

import pytest
import yaml
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException

from src.api import errors, main, schemas
from src.db import connection

CONTRACT = {
    "version": "test-a1",
    "features": [
        {"name": "loan_amnt", "type": "numeric", "min": 1000, "max": 40000},
        {"name": "annual_inc", "type": "numeric", "min": 0},
        {"name": "home_ownership", "type": "categorical", "allowed": ["RENT", "OWN", "MORTGAGE"]},
        {"name": "dti", "type": "numeric", "required": False},
    ],
}
VALID = {"loan_amnt": 12000, "annual_inc": 55000, "home_ownership": "RENT"}
EXAMPLES = {"p": VALID}
SECRET_VALUE = "Jean-Dupont-0612345678"


def write_files(tmp_path, contract=CONTRACT, examples=EXAMPLES):
    contract_path = tmp_path / "features.yaml"
    examples_path = tmp_path / "examples.json"
    contract_path.write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    examples_path.write_text(json.dumps(examples), encoding="utf-8")
    return contract_path, examples_path


def build(tmp_path, contract=CONTRACT, examples=EXAMPLES, database_check=lambda: True):
    contract_path, examples_path = write_files(tmp_path, contract, examples)
    return main.create_app(contract_path, examples_path, database_check=database_check)


def asgi_call(app, path, chunks, headers=(), method="POST"):
    """Appel ASGI direct : le corps arrive en plusieurs messages http.request, comme avec uvicorn
    (le TestClient, lui, le livre en un seul message)."""

    async def run():
        messages = [{"type": "http.request", "body": c, "more_body": i < len(chunks) - 1} for i, c in enumerate(chunks)]
        sent = []

        async def receive():
            if messages:
                await asyncio.sleep(0.005)
                return messages.pop(0)
            await asyncio.sleep(0.5)
            return {"type": "http.disconnect"}

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method,
            "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
            "headers": [(b"content-type", b"application/json"), *headers],
            "client": ("test", 1), "server": ("test", 80),
        }
        await app(scope, receive, send)
        return sent

    return asyncio.run(run())


def statuses(sent):
    return [m["status"] for m in sent if m["type"] == "http.response.start"]


# --- taille du corps -------------------------------------------------------------------------


def _body_of_size(size: int) -> bytes:
    head = json.dumps({**VALID, "padding": ""})[:-2]  # ... "padding": "
    body = head.encode() + b"x" * (size - len(head) - 2) + b'"}'
    assert len(body) == size
    return body


# [E01, E02]
def test_body_of_exactly_the_maximum_size_is_accepted(tmp_path):
    client = TestClient(build(tmp_path))
    headers = {"content-type": "application/json"}
    at_limit = client.post("/predict", content=_body_of_size(errors.MAX_BODY_BYTES), headers=headers)
    assert at_limit.status_code == 422  # champ padding inconnu : le corps a été lu et validé
    over = client.post("/predict", content=_body_of_size(errors.MAX_BODY_BYTES + 1), headers=headers)
    assert over.status_code == 413


# [E03]
def test_declared_content_length_is_refused_even_if_the_route_ignores_the_body(tmp_path):
    client = TestClient(build(tmp_path))
    response = client.request("GET", "/health", content=b"x" * (errors.MAX_BODY_BYTES + 1))
    assert response.status_code == 413
    assert response.json()["code"] == "PAYLOAD_TOO_LARGE"


# [E05]
def test_body_sent_in_many_small_messages_is_counted_cumulatively(tmp_path):
    app = build(tmp_path)
    chunks = [b'{"padding": "'] + [b"x" * 1024] * 80 + [b'"}']
    assert statuses(asgi_call(app, "/predict", chunks)) == [413]


# [E08]
def test_body_overflow_after_a_streaming_response_started_sends_no_second_response(tmp_path):
    app = build(tmp_path)

    @app.post("/essai-flux")
    def streaming():
        async def chunks():
            for _ in range(300):  # 6 s au plus : le dépassement du corps arrive bien avant la fin
                yield b"a"
                await asyncio.sleep(0.02)

        return StreamingResponse(chunks())

    sent = asgi_call(app, "/essai-flux", [b"x" * 1024] * 80)
    assert statuses(sent) == [200]  # pas de seconde ligne de statut (413) au milieu du flux
    assert sent[-1].get("more_body") is True  # le flux a bien été interrompu par le dépassement


# [E25]
def test_non_numeric_content_length_is_ignored(tmp_path):
    app = build(tmp_path)
    sent = asgi_call(app, "/predict", [json.dumps(VALID).encode()], headers=[(b"content-length", b"abc")])
    assert statuses(sent) == [503]  # validation passée, sans champion


# [E23]
def test_http_exception_413_keeps_the_payload_too_large_code(tmp_path):
    app = build(tmp_path)

    @app.get("/essai-413")
    def too_large():
        raise HTTPException(status_code=413)

    body = TestClient(app).get("/essai-413").json()
    assert body["code"] == "PAYLOAD_TOO_LARGE"


# --- journal ---------------------------------------------------------------------------------


# [E15]
def test_head_requests_are_logged_with_their_method(tmp_path, caplog):
    client = TestClient(build(tmp_path))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        client.head("/health")
    assert "HEAD /health -> 200" in caplog.text


# [E28, E30]
def test_custom_route_methods_outside_the_known_list_are_not_logged(tmp_path, caplog):
    app = build(tmp_path)
    method = SECRET_VALUE.upper().replace("-", "_")

    def crash(payload: schemas.TrainRequest):
        raise ValueError("imprévu")

    app.add_api_route("/essai-methode", crash, methods=[method])
    client = TestClient(app)
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert client.request(method, "/essai-methode", json={"preset": "lent"}).status_code == 422
        assert client.request(method, "/essai-methode", json={"preset": "rapide"}).status_code == 500
    assert method not in caplog.text
    assert "Demande refusée (422) sur <méthode inconnue> /essai-methode" in caplog.text
    assert "Erreur interne sur <méthode inconnue> /essai-methode" in caplog.text


# [E20]
def test_exactly_the_maximum_number_of_errors_is_not_called_truncated(tmp_path):
    client = TestClient(build(tmp_path))
    body = {**VALID, **{f"inconnu_{i}": 1 for i in range(errors.MAX_FIELD_ERRORS)}}
    response = client.post("/predict", json=body).json()
    assert len(response["errors"]) == errors.MAX_FIELD_ERRORS
    assert "premières" not in response["detail"]


# [E22]
def test_validation_log_names_at_most_the_maximum_number_of_contract_fields(tmp_path, caplog):
    count = errors.MAX_FIELD_ERRORS + 10
    contract = {"features": [{"name": f"v{i:03d}", "type": "numeric"} for i in range(count)]}
    examples = {"p": {f"v{i:03d}": 1.0 for i in range(count)}}
    client = TestClient(build(tmp_path, contract=contract, examples=examples))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert client.post("/predict", json={}).status_code == 422
    (line,) = [r.getMessage() for r in caplog.records if "Demande refusée" in r.getMessage()]
    assert line.count("(missing)") == errors.MAX_FIELD_ERRORS


# [E31]
@pytest.mark.parametrize("root", ["query", "path", "header", "cookie"])
def test_field_names_drop_every_location_root(root):
    assert errors.field_name((root, "limite")) == "limite"


# --- /health ---------------------------------------------------------------------------------


# [M01]
def test_a_hanging_database_check_does_not_start_more_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "HEALTH_CHECK_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()
    calls = []

    def hanging():
        calls.append(1)
        release.wait(5)
        return True

    client = TestClient(build(tmp_path, database_check=hanging))
    try:
        for _ in range(3):
            assert client.get("/health").json()["database"] == "unavailable"
        assert len(calls) == 1
    finally:
        release.set()


# [M04]
def test_health_waits_at_most_two_seconds_without_monkeypatch(tmp_path):
    release = threading.Event()
    client = TestClient(build(tmp_path, database_check=lambda: release.wait(10)))
    try:
        started = time.perf_counter()
        assert client.get("/health").json()["database"] == "unavailable"
        assert time.perf_counter() - started < 2.6
    finally:
        release.set()


# [M10]
def test_head_health_is_not_published_in_openapi(tmp_path):
    operations = TestClient(build(tmp_path)).get("/openapi.json").json()["paths"]["/health"]
    assert set(operations) == {"get"}


# --- démarrage par uvicorn -------------------------------------------------------------------


# [M15]
def test_module_app_with_broken_contract_exits_with_code_1(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "CONTRACT_PATH", tmp_path / "absent.yaml")
    monkeypatch.delitem(vars(main), "app", raising=False)
    with pytest.raises(SystemExit) as caught:
        main.app  # noqa: B018
    assert caught.value.code == 1


# [M17]
def test_unknown_module_attribute_raises_attribute_error():
    with pytest.raises(AttributeError):
        main.inexistant  # noqa: B018
    assert not hasattr(main, "inexistant")


# --- contrat et exemples ---------------------------------------------------------------------


# [M22]
def test_boolean_contract_version_stops_startup(tmp_path):
    contract_path, examples_path = write_files(tmp_path, contract={**CONTRACT, "version": True})
    with pytest.raises(main.ContractError, match="version doit être un texte"):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


# [M24]
def test_pydantic_schema_errors_become_contract_errors(tmp_path, monkeypatch):
    # Filet de main._build_request_model : une erreur imprévue de Pydantic (ici simulée, les
    # cas connus étant refusés plus tôt par une ValueError) devient une ContractError.
    def unexpected(contract):
        raise TypeError("cas non prévu")

    monkeypatch.setattr(main, "build_request_model", unexpected)
    contract_path, examples_path = write_files(tmp_path)
    with pytest.raises(main.ContractError, match=r"features\.yaml.*TypeError"):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


# [M26]
def test_non_object_example_is_named_as_such(tmp_path):
    contract_path, examples_path = write_files(tmp_path, examples={"risque": [1, 2]})
    with pytest.raises(main.ContractError, match="« risque » invalide.*un objet JSON est attendu"):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


# [M29]
def test_hanging_database_check_is_logged_as_too_long(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(main, "HEALTH_CHECK_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()
    client = TestClient(build(tmp_path, database_check=lambda: release.wait(5)))
    try:
        with caplog.at_level(logging.DEBUG, logger="neuraxion"):
            client.get("/health")
        assert "Contrôle de la base trop long" in caplog.text
    finally:
        release.set()


# --- schemas.py ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("feature", "message"),
    [
        ({"name": "home", "type": "categorical", "allowed": []}, "allowed doit être une liste non vide"),  # [S07]
        ({"name": "term", "type": "integer", "allowed": [36, 60.5]}, "ne correspond pas au type integer"),  # [S09]
        ({"name": "term", "type": "integer", "allowed": [36, True]}, "ne correspond pas au type integer"),
        ({"name": "x", "type": "numeric", "allowed": [1, False]}, "ne correspond pas au type numeric"),
    ],
)
def test_request_factory_names_the_allowed_problem(feature, message):
    with pytest.raises(ValueError, match=message):
        schemas.build_request_model({"features": [feature]})


# [S27]
def test_non_mapping_feature_is_named_by_position():
    with pytest.raises(ValueError, match="n° 1 du contrat sans name ou sans type"):
        schemas.build_request_model({"features": ["loan_amnt"]})


# [S12]
def test_min_equal_to_max_is_a_valid_contract():
    model = schemas.build_request_model({"features": [{"name": "x", "type": "integer", "min": 5, "max": 5}]})
    assert model.model_validate({"x": 5}).x == 5


# [S28, S29]
@pytest.mark.parametrize("model_info", [{"name": "m", "version": "3", "decision_threshold": 1.5},
                                        {"name": "m", "version": "champion", "decision_threshold": 0.3}])
def test_health_rejects_an_out_of_contract_model_info(tmp_path, model_info):
    app = build(tmp_path)
    app.state.model_info = model_info
    assert TestClient(app).get("/health").json() == {"status": "degraded", "database": "ok", "model": None}


# --- connection.py ---------------------------------------------------------------------------


# [C06]
def test_database_is_up_swallows_any_driver_error(monkeypatch):
    def broken(**kwargs):
        raise RuntimeError("pilote cassé")

    monkeypatch.setattr(connection.psycopg, "connect", broken)
    env = {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d"}
    assert connection.database_is_up(env) is False


# [S03] Pydantic ferait de _prive un attribut privé : la variable disparaîtrait du modèle.
@pytest.mark.parametrize("name", ["_prive", "1x", "a-b", "é"])
def test_invalid_feature_names_are_refused_by_the_factory(name):
    with pytest.raises(ValueError, match="Nom de variable invalide"):
        schemas.build_request_model({"features": [{"name": name, "type": "numeric"}]})

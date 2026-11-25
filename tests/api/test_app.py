"""Tâche A1 : application, /health, GET /schema, erreurs au format commun et logs.

Aucun test ne demande PostgreSQL ni MLflow : la base est simulée par database_check.
"""

import asyncio
import json
import logging
import re
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, field_validator
from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from starlette.exceptions import HTTPException

from src.api import errors, main, schemas
from src.api.errors import ApiError, french_message
from src.db import connection

CONTRACT = {
    "version": "test-a1",
    "features": [
        {"name": "loan_amnt", "type": "numeric", "min": 1000, "max": 40000, "label": "Montant", "unit": "dollars"},
        {"name": "annual_inc", "type": "numeric", "min": 0},
        {"name": "open_acc", "type": "integer", "min": 0, "required": False},
        {"name": "home_ownership", "type": "categorical", "allowed": ["RENT", "OWN", "MORTGAGE"]},
        {"name": "dti", "type": "numeric", "required": False},
    ],
}
EXAMPLES = {
    "prudent": {"loan_amnt": 5000, "annual_inc": 90000, "open_acc": 8, "home_ownership": "MORTGAGE"},
    "risque": {"loan_amnt": 35000, "annual_inc": 30000, "home_ownership": "RENT"},
}
VALID = {"loan_amnt": 12000, "annual_inc": 55000, "home_ownership": "RENT"}
SECRET_VALUE = "Jean-Dupont-0612345678"
REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_TEXT = (REPO_ROOT / "docs" / "api.md").read_text(encoding="utf-8")


def doc_contract() -> dict:
    (block,) = re.findall(r"<!-- contrat-illustration -->\s*```yaml\s*(.*?)```", DOC_TEXT, re.DOTALL)
    return yaml.safe_load(block)


def doc_examples() -> dict:
    found = re.findall(r"<!-- exemple: (\w+) -->\s*```json\s*(.*?)```", DOC_TEXT, re.DOTALL)
    examples: dict = {}
    for name, body in found:
        examples.setdefault(name, json.loads(body))  # le premier exemple de chaque modèle
    return examples


def write_files(tmp_path, contract=CONTRACT, examples=EXAMPLES):
    contract_path = tmp_path / "features.yaml"
    examples_path = tmp_path / "examples.json"
    contract_path.write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    examples_path.write_text(json.dumps(examples), encoding="utf-8")
    return contract_path, examples_path


@pytest.fixture
def make_app(tmp_path):
    def factory(database_up=True, contract=CONTRACT, examples=EXAMPLES):
        contract_path, examples_path = write_files(tmp_path, contract, examples)
        return main.create_app(contract_path, examples_path, database_check=lambda: database_up)

    return factory


@pytest.fixture
def client(make_app):
    return TestClient(make_app())


MODEL = schemas.ModelInfo(name="credit-default", version="3", decision_threshold=0.31)


# --- /health ---------------------------------------------------------------


def test_health_is_degraded_without_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "database": "ok", "model": None}


def test_health_is_ok_with_database_and_model(make_app):
    app = make_app()
    app.state.model_info = MODEL
    body = TestClient(app).get("/health").json()
    assert body == {"status": "ok", "database": "ok", "model": MODEL.model_dump()}


def test_health_reports_unavailable_database_with_status_200(make_app):
    app = make_app(database_up=False)
    app.state.model_info = MODEL
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["database"] == "unavailable"


def test_health_survives_a_failing_database_check(make_app):
    app = make_app()

    def broken():
        raise OSError("connexion refusée")

    app.state.database_check = broken
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["database"] == "unavailable"


def test_health_logs_database_state_changes_once(make_app, caplog):
    app = make_app(database_up=False)
    client = TestClient(app)
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        for _ in range(3):
            client.get("/health")
    warnings = [r for r in caplog.records if "PostgreSQL ne répond pas" in r.getMessage()]
    assert len(warnings) == 1


# --- /schema ---------------------------------------------------------------


def test_schema_returns_the_validation_schema_and_examples(client):
    response = client.get("/schema")
    assert response.status_code == 200
    body = schemas.SchemaResponse.model_validate(response.json())
    assert body.contract_version == "test-a1"
    assert body.request_schema == schemas.build_request_model(CONTRACT).model_json_schema()
    assert body.examples == EXAMPLES
    assert body.request_schema["additionalProperties"] is False


def test_schema_examples_are_accepted_by_predict(client):
    examples = client.get("/schema").json()["examples"]
    for example in examples.values():
        # La validation passe : la route vide répond 501, pas 422.
        assert client.post("/predict", json=example).status_code == 501


def test_schema_reports_missing_contract_version(make_app):
    contract = {key: value for key, value in CONTRACT.items() if key != "version"}
    body = TestClient(make_app(contract=contract)).get("/schema").json()
    assert body["contract_version"] == main.UNSPECIFIED_VERSION


def test_openapi_lists_all_routes(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/schema", "/predict", "/train", "/feedback"} <= set(paths)


# --- 422 au format commun --------------------------------------------------


def test_unknown_field_gets_a_clear_422(client):
    response = client.post("/predict", json={**VALID, "nickname": "x"})
    assert response.status_code == 422
    body = schemas.ErrorResponse.model_validate(response.json())
    assert body.code == schemas.ErrorCode.VALIDATION_ERROR
    assert [(e.field, e.message) for e in body.errors] == [("nickname", "champ inconnu")]
    assert "GET /schema" in body.detail


@pytest.mark.parametrize(
    ("change", "field", "message"),
    [
        ({"loan_amnt": 500}, "loan_amnt", "doit être supérieur ou égal à 1000"),
        ({"loan_amnt": 40001}, "loan_amnt", "doit être inférieur ou égal à 40000"),
        ({"annual_inc": "55000"}, "annual_inc", "doit être un nombre"),
        ({"annual_inc": True}, "annual_inc", "doit être un nombre"),
        ({"open_acc": 2.5}, "open_acc", "doit être un nombre entier"),
        ({"home_ownership": "BOAT"}, "home_ownership", "valeur non permise, valeurs possibles : 'RENT', 'OWN' ou 'MORTGAGE'"),
    ],
)
def test_invalid_values_get_french_messages(client, change, field, message):
    response = client.post("/predict", json={**VALID, **change})
    assert response.status_code == 422
    assert response.json()["errors"] == [{"field": field, "message": message}]


def test_missing_field_is_named(client):
    body = client.post("/predict", json={"loan_amnt": 12000, "home_ownership": "RENT"}).json()
    assert body["errors"] == [{"field": "annual_inc", "message": "champ obligatoire manquant"}]


def test_several_errors_are_all_reported(client):
    body = client.post("/predict", json={"loan_amnt": 1, "nickname": "x"}).json()
    assert {e["field"] for e in body["errors"]} == {"loan_amnt", "annual_inc", "home_ownership", "nickname"}
    assert "4 champs" in body["detail"]


@pytest.mark.parametrize("raw", ["{pas du json", "[1, 2]", '"texte"'])
def test_malformed_bodies_get_the_common_format(client, raw):
    response = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422
    body = schemas.ErrorResponse.model_validate(response.json())
    assert body.errors[0].field == "corps"


@pytest.mark.parametrize("value", ["Infinity", "-Infinity", "NaN", "1e400"])
def test_non_finite_numbers_are_refused_even_without_bounds(client, value):
    # dti n'a pas de bornes : seul allow_inf_nan peut refuser ces valeurs.
    raw = '{"loan_amnt": 12000, "annual_inc": 1, "home_ownership": "RENT", "dti": %s}' % value
    response = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["errors"] == [{"field": "dti", "message": "doit être un nombre fini"}]


def test_validation_errors_never_echo_the_submitted_values(client):
    payloads = [
        {**VALID, "home_ownership": SECRET_VALUE},
        {**VALID, "annual_inc": SECRET_VALUE},
        {**VALID, "loan_amnt": SECRET_VALUE},
    ]
    for payload in payloads:
        response = client.post("/predict", json=payload)
        assert response.status_code == 422
        assert SECRET_VALUE not in response.text


def test_train_preset_errors_use_the_common_format(client):
    body = client.post("/train", json={"preset": "lent"}).json()
    assert body["code"] == "VALIDATION_ERROR"
    assert body["errors"][0]["field"] == "preset"


# --- autres erreurs --------------------------------------------------------


def test_unknown_route_gets_404_in_the_common_format(client):
    response = client.get("/nulle-part")
    assert response.status_code == 404
    assert schemas.ErrorResponse.model_validate(response.json()).code == schemas.ErrorCode.NOT_FOUND


def test_wrong_method_gets_405_in_the_common_format(client):
    response = client.get("/predict")
    assert response.status_code == 405
    assert response.json()["code"] == "METHOD_NOT_ALLOWED"
    assert "POST" in response.headers["allow"]


def test_api_error_is_returned_as_raised(make_app):
    app = make_app()

    @app.get("/essai-503")
    def unavailable():
        raise ApiError(503, schemas.ErrorCode.MODEL_UNAVAILABLE, "Aucun modèle validé n'est disponible.")

    response = TestClient(app).get("/essai-503")
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Aucun modèle validé n'est disponible.",
        "code": "MODEL_UNAVAILABLE",
        "errors": None,
    }


def test_api_error_keeps_its_headers(make_app):
    app = make_app()

    @app.get("/essai-401")
    def unauthorized():
        raise ApiError(401, schemas.ErrorCode.UNAUTHORIZED, "Clé absente.", headers={"WWW-Authenticate": "ApiKey"})

    response = TestClient(app).get("/essai-401")
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "ApiKey"


def test_unexpected_error_gets_500_without_its_message(make_app, caplog):
    app = make_app()

    @app.get("/essai-500")
    def crash():
        raise ValueError(f"valeur reçue : {SECRET_VALUE}")

    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = TestClient(app).get("/essai-500")
    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert SECRET_VALUE not in response.text
    assert SECRET_VALUE not in caplog.text
    assert "ValueError" in caplog.text


def test_existing_stub_routes_still_answer_501(client):
    feedback = {"inference_id": "6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d", "feedback": "agree"}
    for route, body in (("/predict", VALID), ("/train", {"preset": "rapide"}), ("/feedback", feedback)):
        response = client.post(route, json=body)
        assert response.status_code == 501
        assert response.json()["code"] == "NOT_IMPLEMENTED"


# --- logs ------------------------------------------------------------------


def test_request_logs_contain_route_status_and_duration_but_no_data(client, caplog):
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        client.post("/predict", json={**VALID, "home_ownership": SECRET_VALUE})
        client.post("/predict", json={**VALID, SECRET_VALUE: 1})
        client.get(f"/inconnue/{SECRET_VALUE}?q={SECRET_VALUE}")
    assert SECRET_VALUE not in caplog.text
    assert "POST /predict -> 422" in caplog.text
    assert "home_ownership (literal_error)" in caplog.text
    assert "autres champs (inconnus ou non déclarés) : 1" in caplog.text
    assert f"GET {'<route inconnue>'} -> 404" in caplog.text
    assert " ms)" in caplog.text


def test_uvicorn_access_log_is_disabled(client):
    # Le journal d'accès d'uvicorn écrit l'adresse IP du client.
    assert logging.getLogger("uvicorn.access").disabled


# --- démarrage -------------------------------------------------------------


def test_missing_contract_stops_startup_with_a_clear_message(tmp_path):
    with pytest.raises(main.ContractError, match="introuvable.*C1"):
        main.create_app(tmp_path / "absent.yaml", tmp_path / "absent.json", database_check=lambda: True)


@pytest.mark.parametrize(
    "contract",
    [
        {"features": []},
        {"features": [{"name": "x", "type": "texte"}]},
        {"features": [{"type": "numeric"}]},
        {"features": ["loan_amnt"]},
        {"features": "loan_amnt"},
        ["loan_amnt"],
        {"features": [{"name": "loan_amnt", "type": "numeric", "min": "1000"}]},
        {"features": [{"name": "loan_amnt", "type": "numeric", "max": [1]}]},
        {"features": [{"name": "loan_amnt", "type": "numeric", "min": True}]},
        {"features": [{"name": "loan_amnt", "type": "numeric", "min": 10, "max": 1}]},
        {"features": [{"name": "grade", "type": "categorical", "min": 1}]},
        {"features": [{"name": "_prive", "type": "numeric"}]},
        {"features": [{"name": "model_x", "type": "numeric"}]},
        {"features": [{"name": "avec espace", "type": "numeric"}]},
        {"features": [{"name": 12, "type": "numeric"}]},
        {"features": [{"name": "home", "type": "categorical", "allowed": "RENT"}]},
        {"features": [{"name": "home", "type": "categorical", "allowed": []}]},
        {"features": [{"name": "term", "type": "integer", "allowed": [36, "60"]}]},
        {"features": [{"name": "term", "type": "integer", "required": "false"}]},
        {"features": [{"name": "term", "type": "integer", "label": 5}]},
        {"version": 1.10, "features": [{"name": "term", "type": "integer"}]},
    ],
)
def test_broken_contract_stops_startup(tmp_path, contract):
    contract_path, examples_path = write_files(tmp_path, contract=contract)
    with pytest.raises(main.ContractError, match="features.yaml"):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


def test_unreadable_yaml_stops_startup_with_a_french_message(tmp_path):
    contract_path, examples_path = write_files(tmp_path)
    contract_path.write_text("features: [\n  {name: secret_extrait\n", encoding="utf-8")
    with pytest.raises(main.ContractError, match=r"syntaxe YAML invalide \(ligne") as caught:
        main.create_app(contract_path, examples_path, database_check=lambda: True)
    assert "secret_extrait" not in str(caught.value)


def test_non_utf8_contract_stops_startup(tmp_path):
    contract_path, examples_path = write_files(tmp_path)
    contract_path.write_bytes("features: [{name: prêt, type: numeric}]".encode("latin-1"))
    with pytest.raises(main.ContractError, match="UTF-8"):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


def test_unreadable_examples_stop_startup_with_a_french_message(tmp_path):
    contract_path, examples_path = write_files(tmp_path)
    examples_path.write_text("{prudent: 1}", encoding="utf-8")
    with pytest.raises(main.ContractError, match=r"JSON invalide \(ligne 1, colonne 2\)"):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


def test_module_app_with_broken_contract_exits_with_one_line(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(main, "CONTRACT_PATH", tmp_path / "absent.yaml")
    monkeypatch.delitem(vars(main), "app", raising=False)
    with caplog.at_level(logging.CRITICAL, logger="neuraxion"), pytest.raises(SystemExit):
        main.app  # noqa: B018 - l'accès construit l'application
    assert "Démarrage impossible : Contrat de données introuvable" in caplog.text


@pytest.mark.parametrize(
    ("examples", "match"),
    [
        ({"risque": {**VALID, "loan_amnt": 1}}, "risque.*loan_amnt"),
        ({"risque": {**VALID, "nickname": "x"}}, "risque.*nickname"),
        ({"risque": [1, 2]}, "risque"),
        ([VALID], "profil"),
        ({}, "profil"),
    ],
)
def test_invalid_examples_stop_startup(tmp_path, examples, match):
    contract_path, examples_path = write_files(tmp_path, examples=examples)
    with pytest.raises(main.ContractError, match=match):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


def test_importing_main_reads_no_file():
    # uvicorn lit src.api.main:app ; importer le module ne doit ni lire contracts/ ni construire l'app.
    code = (
        "import pathlib\n"
        "def interdit(*args, **kwargs):\n"
        "    raise AssertionError('lecture de fichier pendant l import')\n"
        "import src.api.main as m\n"
        "pathlib.Path.read_text = interdit\n"
        "pathlib.Path.is_file = interdit\n"
        "import importlib\n"
        "importlib.reload(m)\n"
        "assert 'app' not in vars(m), 'app construite a l import'\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_module_app_is_built_from_the_default_paths(monkeypatch, tmp_path):
    contract_path, examples_path = write_files(tmp_path)
    monkeypatch.setattr(main, "CONTRACT_PATH", contract_path)
    monkeypatch.setattr(main, "EXAMPLES_PATH", examples_path)
    vars(main).pop("app", None)
    try:
        app = main.app
        assert app is main.app
        assert TestClient(app).get("/schema").json()["contract_version"] == "test-a1"
    finally:
        # Ne pas laisser une application construite sur un contrat temporaire.
        vars(main).pop("app", None)


# --- connexion PostgreSQL --------------------------------------------------


def test_database_is_not_contacted_without_configuration(monkeypatch):
    # Pas d'exception dans le faux connect : database_is_up l'avalerait.
    calls = []
    monkeypatch.setattr(connection.psycopg, "connect", lambda **kwargs: calls.append(kwargs))
    assert connection.database_is_up({"POSTGRES_USER": "u", "POSTGRES_DB": "d"}) is False
    assert calls == []


def test_connection_params_use_defaults_and_timeout():
    params = connection.connection_params({"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d"})
    assert params["host"] == "localhost"
    assert params["port"] == 5432
    assert params["connect_timeout"] == connection.CONNECT_TIMEOUT_SECONDS


def test_database_is_up_returns_false_when_connection_fails(monkeypatch, caplog):
    def refused(**kwargs):
        raise connection.psycopg.OperationalError(f"échec pour le mot de passe {kwargs['password']}")

    monkeypatch.setattr(connection.psycopg, "connect", refused)
    env = {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": SECRET_VALUE, "POSTGRES_DB": "d"}
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        assert connection.database_is_up(env) is False
    assert SECRET_VALUE not in caplog.text


def test_database_is_up_runs_select_1(monkeypatch):
    executed = []

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, query):
            executed.append(query)

    monkeypatch.setattr(connection.psycopg, "connect", lambda **kwargs: FakeConnection())
    env = {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d"}
    assert connection.database_is_up(env) is True
    assert executed == ["SELECT 1"]


# --- docs/api.md -----------------------------------------------------------


def test_doc_validation_example_is_what_the_api_returns(tmp_path):
    contract_path, examples_path = write_files(tmp_path, contract=doc_contract(), examples=doc_examples()["SchemaResponse"]["examples"])
    client = TestClient(main.create_app(contract_path, examples_path, database_check=lambda: True))
    request = next(iter(doc_examples()["SchemaResponse"]["examples"].values()))
    response = client.post("/predict", json={**request, "annual_inc": -1, "nickname": "x"})
    assert response.json() == doc_examples()["ErrorResponse"]


# --- corrections de la boucle 1 ---------------------------------------------


def test_integer_with_allowed_values_stays_strict(tmp_path):
    contract = {"version": "t", "features": [{"name": "term", "type": "integer", "allowed": [36, 60]}]}
    contract_path, examples_path = write_files(tmp_path, contract=contract, examples={"court": {"term": 36}})
    client = TestClient(main.create_app(contract_path, examples_path, database_check=lambda: True))
    cases = {
        "36.0": "doit être un nombre entier",
        "true": "doit être un nombre entier",
        '"36"': "doit être un nombre entier",
        "48": "valeur non permise, valeurs possibles : 36, 60",
    }
    for raw, message in cases.items():
        response = client.post("/predict", content='{"term": %s}' % raw, headers={"content-type": "application/json"})
        assert response.json()["errors"] == [{"field": "term", "message": message}], raw
    assert client.post("/predict", json={"term": 36}).status_code == 501
    published = client.get("/schema").json()["request_schema"]["properties"]["term"]
    assert published["enum"] == [36, 60]
    assert published["type"] == "integer"


@pytest.mark.parametrize("result", ["oui", 1, object(), None])
def test_health_counts_only_true_as_a_working_database(make_app, result):
    app = make_app()
    app.state.database_check = lambda: result
    assert TestClient(app).get("/health").json()["database"] == "unavailable"


def test_health_answers_within_the_timeout_when_the_check_hangs(make_app, monkeypatch):
    monkeypatch.setattr(main, "HEALTH_CHECK_TIMEOUT_SECONDS", 0.3)
    app = make_app()
    app.state.database_check = lambda: time.sleep(2) or True
    started = time.perf_counter()
    response = TestClient(app).get("/health")
    assert time.perf_counter() - started < 1.5
    assert response.status_code == 200
    assert response.json()["database"] == "unavailable"


@pytest.mark.parametrize(
    "model_info",
    ["credit-default", {"name": "credit-default", "version": "champion", "decision_threshold": 0.3}, 42],
)
def test_health_survives_a_malformed_model_info(make_app, model_info, caplog):
    app = make_app()
    app.state.model_info = model_info
    client = TestClient(app)
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        for _ in range(2):
            response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "database": "ok", "model": None}
    assert caplog.text.count("model_info est mal formé") == 1


def test_health_accepts_a_model_object_with_the_right_attributes(make_app):
    class Served:
        name = "credit-default"
        version = "4"
        decision_threshold = 0.29

    app = make_app()
    app.state.model_info = Served()
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_asynchronous_database_check_is_refused(tmp_path):
    async def check():
        return True

    contract_path, examples_path = write_files(tmp_path)
    with pytest.raises(TypeError):
        main.create_app(contract_path, examples_path, database_check=check)


def test_head_health_answers_200(client):
    assert client.head("/health").status_code == 200


def test_invented_http_method_is_not_logged(client, caplog):
    method = SECRET_VALUE.upper().replace("-", "_")
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = client.request(method, "/predict")
    assert response.status_code == 405
    assert method not in caplog.text
    assert "<méthode inconnue> /predict -> 405" in caplog.text


def test_errors_after_the_response_started_are_caught_without_their_message(make_app, caplog):
    app = make_app()

    def fail():
        raise ValueError(f"écriture de {SECRET_VALUE}")

    @app.get("/essai-arriere-plan")
    def with_background(tasks: BackgroundTasks):
        tasks.add_task(fail)
        return {"ok": True}

    @app.get("/essai-flux")
    def streaming():
        def chunks():
            yield b"debut"
            raise ValueError(f"flux de {SECRET_VALUE}")

        return StreamingResponse(chunks())

    client = TestClient(app)
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert client.get("/essai-arriere-plan").status_code == 200
        client.get("/essai-flux")
    assert SECRET_VALUE not in caplog.text
    assert caplog.text.count("ValueError") == 2


def test_internal_error_log_keeps_only_frames_from_src(make_app, caplog):
    app = make_app()

    @app.get("/essai-pile")
    def crash():
        main._served_model(None)  # AttributeError levée dans src/api/main.py

    with caplog.at_level(logging.ERROR, logger="neuraxion"):
        assert TestClient(app).get("/essai-pile").status_code == 500
    assert "src/api/main.py" in caplog.text
    assert "site-packages" not in caplog.text
    assert "starlette" not in caplog.text


def test_too_large_body_is_refused_with_413(client):
    big = {**VALID, "padding": "x" * (70 * 1024)}
    response = client.post("/predict", json=big)
    assert response.status_code == 413
    assert response.json()["code"] == "PAYLOAD_TOO_LARGE"


def test_too_large_chunked_body_is_refused_with_413(client):
    def chunks():
        yield b'{"padding": "'
        for _ in range(80):
            yield b"x" * 1024
        yield b'"}'

    response = client.post("/predict", content=chunks(), headers={"content-type": "application/json"})
    assert response.status_code == 413


def test_field_errors_are_capped(client):
    body = {**VALID, **{f"inconnu_{i}": 1 for i in range(120)}}
    response = client.post("/predict", json=body).json()
    assert len(response["errors"]) == 50
    assert "120 champs" in response["detail"]
    assert "50 premières" in response["detail"]


def test_access_log_is_disabled_again_at_startup(make_app):
    app = make_app()
    logging.getLogger("uvicorn.access").disabled = False  # comme après le dictConfig de uvicorn.run
    with TestClient(app):
        assert logging.getLogger("uvicorn.access").disabled


def test_missing_content_type_gets_a_helpful_message(client):
    response = client.post("/predict", content=json.dumps(VALID))
    assert response.status_code == 422
    assert "Content-Type: application/json" in response.json()["errors"][0]["message"]


def test_lone_surrogate_gets_a_french_message(client):
    raw = '{"loan_amnt": 12000, "annual_inc": 1, "home_ownership": "\\ud800"}'
    response = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert all("Input should" not in error["message"] for error in response.json()["errors"])


def test_unknown_pydantic_error_types_fall_back_to_french():
    assert french_message({"type": "type_inconnu", "msg": "Input should be something"}) == "valeur invalide"


def test_router_lifespan_can_fill_model_info(tmp_path, monkeypatch):
    # Raccord A2 : un lifespan posé sur le routeur predict remplit app.state.model_info.
    original = main.predict.build_router

    def build_router_with_lifespan(request_model):
        @asynccontextmanager
        async def lifespan(application):
            application.state.model_info = MODEL
            yield

        router = APIRouter(lifespan=lifespan)
        router.include_router(original(request_model))
        return router

    monkeypatch.setattr(main.predict, "build_router", build_router_with_lifespan)
    contract_path, examples_path = write_files(tmp_path)
    app = main.create_app(contract_path, examples_path, database_check=lambda: True)
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"]["version"] == "3"


# --- tests proposés par l'injection d'erreurs (boucle 1) -------------------


# P1 (M07) ------------------------------------------------------------------
def test_failing_database_check_message_is_not_logged(make_app, caplog):
    app = make_app()

    def broken():
        raise OSError(f"connexion refusée pour {SECRET_VALUE}")

    app.state.database_check = broken
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        assert TestClient(app).get("/health").json()["database"] == "unavailable"
    assert SECRET_VALUE not in caplog.text
    assert "OSError" in caplog.text


# P2 (M10) ------------------------------------------------------------------
def test_missing_examples_stop_startup_with_the_file_name(tmp_path):
    contract_path, examples_path = write_files(tmp_path)
    examples_path.unlink()
    with pytest.raises(main.ContractError, match=r"introuvables.*examples\.json.*C1"):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


# P3 (M11) ------------------------------------------------------------------
def test_unreadable_yaml_message_names_the_file(tmp_path):
    contract_path, examples_path = write_files(tmp_path)
    contract_path.write_text("features: [\n", encoding="utf-8")
    with pytest.raises(main.ContractError, match=r"illisible.*features\.yaml"):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


# P4 (E05, E06, E27) ----------------------------------------------------------
@pytest.mark.parametrize(
    ("status", "code", "detail"),
    [
        (400, "BAD_REQUEST", "Requête mal formée."),
        (401, "UNAUTHORIZED", "Authentification requise."),
        (403, "FORBIDDEN", "Accès refusé."),
        (404, "NOT_FOUND", "Route inconnue. La liste des routes est dans /docs."),
        (409, "BAD_REQUEST", None),  # TRAINING_IN_PROGRESS seulement par ApiError (A3)
        (500, "INTERNAL_ERROR", None),
        (503, "INTERNAL_ERROR", None),
    ],
)
def test_http_exceptions_are_mapped_to_the_documented_codes(make_app, status, code, detail):
    app = make_app()

    @app.get("/essai-http")
    def raise_http():
        raise HTTPException(status_code=status, detail="détail technique")

    body = TestClient(app).get("/essai-http").json()
    assert body["code"] == code
    assert body["detail"] == (detail or "détail technique")


def test_unknown_route_and_wrong_method_have_french_details(client):
    assert client.get("/nulle-part").json()["detail"] == "Route inconnue. La liste des routes est dans /docs."
    assert client.get("/predict").json()["detail"] == "Méthode non autorisée pour cette route."


# P5 (E17, E14) : règles atteintes seulement par un futur contrat ou une future route ----
def test_fallback_message_never_echoes_the_input():
    error = {"type": "string_too_long", "msg": "String should have at most 3 characters", "input": SECRET_VALUE}
    assert french_message(error) == "valeur invalide"


@pytest.mark.parametrize(
    ("kind", "key", "expected"),
    [("greater_than", "gt", "doit être strictement supérieur à 0"), ("less_than", "lt", "doit être strictement inférieur à 1")],
)
def test_strict_bounds_have_french_messages(kind, key, expected):
    assert french_message({"type": kind, "ctx": {key: 0 if key == "gt" else 1.0}}) == expected


# P6 (E19, E22) -------------------------------------------------------------
def test_logs_use_the_route_template_for_422_and_500(make_app, caplog):
    app = make_app()

    @app.post("/essai/{code}")
    def with_path(code: str, payload: schemas.TrainRequest):
        raise ValueError("imprévu")

    client = TestClient(app)
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        client.post(f"/essai/{SECRET_VALUE}", json={"preset": "lent"})
        client.post(f"/essai/{SECRET_VALUE}", json={"preset": "rapide"})
    assert SECRET_VALUE not in caplog.text
    assert "Demande refusée (422) sur POST /essai/{code}" in caplog.text
    assert "Erreur interne sur POST /essai/{code}" in caplog.text


# P7 (S03, E29) -------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "field"),
    [
        ('{"loan_amnt": Infinity, "annual_inc": 1, "home_ownership": "RENT"}', "loan_amnt"),
        ('{"loan_amnt": 12000, "annual_inc": NaN, "home_ownership": "RENT"}', "annual_inc"),
    ],
)
def test_non_finite_numbers_get_their_own_message(client, raw, field):
    response = client.post("/predict", content=raw, headers={"content-type": "application/json"})
    assert response.json()["errors"] == [{"field": field, "message": "doit être un nombre fini"}]


# P8 (S06) ------------------------------------------------------------------
def test_integer_bounds_are_enforced(client):
    response = client.post("/predict", json={**VALID, "open_acc": -1})
    assert response.json()["errors"] == [{"field": "open_acc", "message": "doit être supérieur ou égal à 0"}]


# P9 (E11) ------------------------------------------------------------------
def test_float_bounds_are_written_without_decimal_part(make_app):
    contract = {"features": [{"name": "loan_amnt", "type": "numeric", "min": 1000.0, "max": 40000.0}]}
    app = make_app(contract=contract, examples={"p": {"loan_amnt": 5000}})
    body = TestClient(app).post("/predict", json={"loan_amnt": 1}).json()
    assert body["errors"] == [{"field": "loan_amnt", "message": "doit être supérieur ou égal à 1000"}]


# P10 (S13, S14) --------------------------------------------------------------
@pytest.mark.parametrize("contract", [{"features": 5}, {"features": True}, ["loan_amnt"], None])
def test_request_factory_raises_value_error_on_malformed_contracts(contract):
    with pytest.raises(ValueError):
        schemas.build_request_model(contract)


# P11 (S17, S18, S20) ---------------------------------------------------------
def test_error_codes_are_the_documented_strings():
    documented = dict(re.findall(r"^\| `([A-Z_]+)` \| (\d{3}) \|", DOC_TEXT, re.MULTILINE))
    assert {code.value for code in schemas.ErrorCode} == set(documented)
    assert all(code.value == code.name for code in schemas.ErrorCode)


# P12 (D01, D02, D07) ---------------------------------------------------------
@pytest.mark.parametrize("missing", ["POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"])
def test_incomplete_configuration_is_refused_without_connection(monkeypatch, missing):
    calls = []
    monkeypatch.setattr(connection.psycopg, "connect", lambda **kwargs: calls.append(kwargs))
    env = {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d", missing: ""}
    with pytest.raises(connection.DatabaseNotConfigured, match=missing):
        connection.connection_params(env)
    assert connection.database_is_up(env) is False
    assert calls == []  # une AssertionError levée dans connect serait avalée par database_is_up


# P13 (D04) -----------------------------------------------------------------
def test_connection_timeout_is_short_and_finite():
    params = connection.connection_params({"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d"})
    assert params["connect_timeout"] == 2  # 0 signifie « attendre indéfiniment » pour libpq


# --- corrections de la boucle 2 ---------------------------------------------


def test_hung_database_check_is_shared_and_not_queued(make_app, monkeypatch):
    monkeypatch.setattr(main, "HEALTH_CHECK_TIMEOUT_SECONDS", 0.2)
    release = threading.Event()
    calls = []

    def hung():
        calls.append(1)
        release.wait(5)
        return True

    app = make_app()
    app.state.database_check = hung
    client = TestClient(app)
    for _ in range(3):
        assert client.get("/health").json()["database"] == "unavailable"
    assert len(calls) == 1  # un seul contrôle lancé, aucun en attente
    release.set()
    app.state.database_future.result(timeout=2)
    assert client.get("/health").json()["database"] == "ok"
    assert len(calls) == 2


def test_concurrent_health_calls_stay_within_the_timeout(make_app, monkeypatch):
    monkeypatch.setattr(main, "HEALTH_CHECK_TIMEOUT_SECONDS", 0.3)
    app = make_app()
    app.state.database_check = lambda: time.sleep(3) or True
    client = TestClient(app)
    durations = []

    def call():
        started = time.perf_counter()
        client.get("/health")
        durations.append(time.perf_counter() - started)

    threads = [threading.Thread(target=call) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert max(durations) < 1.5


def test_hung_database_check_does_not_block_process_exit(tmp_path):
    contract_path, examples_path = write_files(tmp_path)
    code = (
        "import threading, sys\n"
        "from fastapi.testclient import TestClient\n"
        "from src.api import main\n"
        "main.HEALTH_CHECK_TIMEOUT_SECONDS = 0.2\n"
        f"app = main.create_app(r'{contract_path}', r'{examples_path}', database_check=lambda: threading.Event().wait())\n"
        "print(TestClient(app).get('/health').json()['database'])\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "unavailable"


def test_unvalidated_model_info_instance_is_revalidated(make_app):
    app = make_app()
    app.state.model_info = schemas.ModelInfo.model_construct(name="credit-default", version="champion", decision_threshold=0.2)
    assert TestClient(app).get("/health").json() == {"status": "degraded", "database": "ok", "model": None}


def test_asynchronous_callable_object_is_refused(tmp_path):
    class AsyncCheck:
        async def __call__(self):
            return True

    contract_path, examples_path = write_files(tmp_path)
    with pytest.raises(TypeError):
        main.create_app(contract_path, examples_path, database_check=AsyncCheck())


def test_check_returning_a_coroutine_is_unavailable_without_warning(make_app, recwarn):
    async def inner():
        return True

    app = make_app()
    app.state.database_check = lambda: inner()
    assert TestClient(app).get("/health").json()["database"] == "unavailable"
    assert not [w for w in recwarn.list if "never awaited" in str(w.message)]


def test_unreadable_contract_file_stops_startup(tmp_path, monkeypatch):
    contract_path, examples_path = write_files(tmp_path)
    original = Path.read_text

    def denied(self, *args, **kwargs):
        if self == contract_path:
            raise PermissionError("accès refusé")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(main.ContractError, match=r"accès impossible \(PermissionError\)"):
        main.create_app(contract_path, examples_path, database_check=lambda: True)


@pytest.mark.parametrize(
    "feature",
    [
        {"name": "score", "type": "integer", "min": 0.5, "max": 2.5},
        {"name": "score\n", "type": "integer"},
        {"name": "score", "type": "integer", "example": "hors-type"},
        {"name": "score", "type": "integer", "min": 0, "example": -1},
        {"name": "score", "type": "integer", "allowed": [1, 9], "max": 3},
        {"name": "score", "type": "integer", "allowed": [1, 1]},
        {"name": "home", "type": "categorical", "allowed": ["RENT"], "example": "BOAT"},
    ],
)
def test_contradictory_contract_metadata_stops_startup(feature):
    with pytest.raises(ValueError):
        schemas.build_request_model({"features": [feature]})


@pytest.mark.parametrize(("kind", "allowed", "json_type"), [("integer", [1, 3], "integer"), ("numeric", [1.25, 2.5], "number")])
def test_optional_numeric_list_publishes_null_like_the_api_accepts_it(tmp_path, kind, allowed, json_type):
    contract = {"features": [{"name": "score", "type": kind, "allowed": allowed, "required": False, "min": 0}]}
    contract_path, examples_path = write_files(tmp_path, contract=contract, examples={"vide": {}})
    client = TestClient(main.create_app(contract_path, examples_path, database_check=lambda: True))
    published = client.get("/schema").json()["request_schema"]["properties"]["score"]
    branches = published["anyOf"]
    assert {"type": "null"} in branches
    assert {"type": json_type, "enum": allowed, "minimum": 0} in branches
    assert "enum" not in published
    assert client.post("/predict", json={"score": None}).status_code == 501
    assert client.post("/predict", json={"score": allowed[0]}).status_code == 501
    assert client.post("/predict", json={"score": 2}).status_code == 422


def test_overflow_stays_413_when_the_downstream_app_fails_differently():
    async def downstream(scope, receive, send):
        try:
            while (await receive()).get("more_body"):
                pass
        except Exception as error:
            raise RuntimeError("échec en aval") from error

    client = TestClient(errors.RequestGuard(downstream, max_body_bytes=10))

    def chunks():
        yield b"x" * 8
        yield b"x" * 8

    response = client.post("/", content=chunks())
    assert response.status_code == 413
    assert response.json()["code"] == "PAYLOAD_TOO_LARGE"


def test_extension_routes_do_not_leak_client_keys_or_validator_values(make_app, caplog):
    class Payload(BaseModel):
        scores: dict[str, int]
        note: str

        @field_validator("note")
        @classmethod
        def check_note(cls, value):
            raise ValueError(f"note refusée : {value}")

    app = make_app()

    @app.post("/essai-extension")
    def extension(payload: Payload):
        return {}

    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = TestClient(app).post("/essai-extension", json={"scores": {SECRET_VALUE: "x"}, "note": SECRET_VALUE})
    assert response.status_code == 422
    assert SECRET_VALUE not in caplog.text
    messages = [error["message"] for error in response.json()["errors"]]
    assert all(SECRET_VALUE not in message for message in messages)
    assert "valeur invalide" in messages


def test_body_level_errors_get_a_body_detail(client):
    body = client.post("/predict", content="{pas du json", headers={"content-type": "application/json"}).json()
    assert "corps de la requête" in body["detail"]
    assert "un champ" not in body["detail"]


# --- corrections de la boucle 3 ---------------------------------------------


def test_database_transition_is_logged_once_under_concurrent_calls(make_app, caplog):
    app = make_app()
    app.state.database_check = lambda: time.sleep(0.2) or True
    client = TestClient(app)
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        threads = [threading.Thread(target=client.get, args=("/health",)) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert caplog.text.count("PostgreSQL répond.") == 1


@pytest.mark.parametrize("error", [RuntimeError(f"échec {SECRET_VALUE}"), asyncio.CancelledError()])
def test_failing_database_check_never_escapes(make_app, error, recwarn, caplog):
    def check():
        raise error

    app = make_app()
    app.state.database_check = check
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["database"] == "unavailable"
    assert SECRET_VALUE not in caplog.text
    assert not [w for w in recwarn.list if "never retrieved" in str(w.message)]


def test_model_info_serializers_are_not_called(make_app):
    class Served(BaseModel):
        name: str = "credit-default"
        version: str = "4"
        decision_threshold: float = 0.3

        def model_dump(self, *args, **kwargs):
            raise ValueError("sérialiseur interdit")

    app = make_app()
    app.state.model_info = Served()
    assert TestClient(app).get("/health").json()["status"] == "ok"


def test_train_validation_log_names_the_declared_preset(client, caplog):
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        client.post("/train", json={"preset": "lent", SECRET_VALUE: 1})
    assert "preset (literal_error)" in caplog.text
    assert "autres champs (inconnus ou non déclarés) : 1" in caplog.text
    assert SECRET_VALUE not in caplog.text


def test_a_variable_named_like_the_body_keeps_a_field_detail(tmp_path):
    contract = {"features": [{"name": "corps", "type": "integer"}]}
    contract_path, examples_path = write_files(tmp_path, contract=contract, examples={"p": {"corps": 1}})
    client = TestClient(main.create_app(contract_path, examples_path, database_check=lambda: True))
    body = client.post("/predict", json={"corps": "x"}).json()
    assert body["errors"] == [{"field": "corps", "message": "doit être un nombre entier"}]
    assert "un champ ne respecte pas le contrat" in body["detail"]


@pytest.mark.parametrize(
    "feature",
    [
        {"name": "score", "type": "numeric", "max": float("inf")},
        {"name": "score", "type": "numeric", "min": float("nan")},
        {"name": "score", "type": "numeric", "allowed": [1, float("nan")]},
        {"name": "score", "type": "numeric", "allowed": [1, float("inf")]},
        {"name": "score", "type": "numeric", "allowed": [1, 9007199254740993]},
    ],
)
def test_non_finite_or_inexact_contract_numbers_are_refused(feature):
    with pytest.raises(ValueError, match="score"):
        schemas.build_request_model({"features": [feature]})


def test_interrupted_streaming_response_is_reported_in_the_logs(caplog):
    async def downstream(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"debut", "more_body": True})
        while (await receive()).get("more_body"):
            pass

    async def run():
        messages = [{"type": "http.request", "body": b"x" * 8, "more_body": True}] * 3
        messages.append({"type": "http.request", "body": b"", "more_body": False})

        async def receive():
            return messages.pop(0)

        sent = []

        async def send(message):
            sent.append(message)

        scope = {"type": "http", "method": "POST", "path": "/flux", "headers": []}
        await errors.RequestGuard(downstream, max_body_bytes=10)(scope, receive, send)
        return sent

    with caplog.at_level(logging.INFO, logger="neuraxion"):
        sent = asyncio.run(run())
    assert [m["type"] for m in sent].count("http.response.start") == 1
    assert "Réponse interrompue" in caplog.text


def test_model_info_with_extra_fields_is_accepted(make_app):
    # Le modèle servi de A2 peut porter d'autres attributs : seuls les trois attendus sont lus.
    class Served(BaseModel):
        name: str = "credit-default"
        version: str = "4"
        decision_threshold: float = 0.3
        run_id: str = "abc"

    app = make_app()
    for value in (Served(), {**Served().model_dump()}):
        app.state.model_info = value
        assert TestClient(app).get("/health").json()["model"] == {
            "name": "credit-default",
            "version": "4",
            "decision_threshold": 0.3,
        }


def test_documentation_routes_are_named_in_the_logs(client, caplog):
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        client.get("/docs")
        client.get("/openapi.json")
    assert "GET /docs -> 200" in caplog.text
    assert "GET /openapi.json -> 200" in caplog.text

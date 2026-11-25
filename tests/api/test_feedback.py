"""Tests de POST /feedback (tâche F1), sans base : l'écriture est remplacée par un faux.

Les mêmes comportements contre un vrai PostgreSQL : tests/db/test_feedback_postgres.py.
"""

import json
import logging
import re
import types
from uuid import UUID

import psycopg
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import feedback, main
from src.api.errors import install_error_handlers
from src.db import inferences, init_db

DOC_PATH = main.ROOT / "docs" / "api.md"

INFERENCE_ID = "6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d"
FREE_TEXT = "le client a payé, secret-7f3a"


class FakeStore:
    """Remplace feedback.store_feedback : garde les appels et renvoie la réponse prévue."""

    def __init__(self, found=True, error=None):
        self.found, self.error, self.calls = found, error, []

    def __call__(self, inference_id, value):
        self.calls.append((inference_id, value))
        if self.error is not None:
            raise self.error
        return self.found


@pytest.fixture
def app() -> FastAPI:
    application = FastAPI()
    install_error_handlers(application)  # comme main.py : la route lève ApiError
    application.include_router(feedback.router)
    return application


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


@pytest.fixture
def store(monkeypatch) -> FakeStore:
    fake = FakeStore()
    monkeypatch.setattr(feedback, "store_feedback", fake)
    return fake


# --- succès -------------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["agree", "disagree"])
def test_feedback_is_recorded_and_echoed(client, store, caplog, value):
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": value})
    assert response.status_code == 200, response.text
    assert response.json() == {"inference_id": INFERENCE_ID, "feedback": value}
    assert store.calls == [(UUID(INFERENCE_ID), value)]
    # Le succès est journalisé sans la valeur choisie, comme la décision de POST /predict.
    assert f"Feedback enregistré pour l'inférence {INFERENCE_ID}" in caplog.text
    assert "agree" not in caplog.text


def test_uppercase_uuid_is_accepted_and_answered_in_canonical_form(client, store):
    response = client.post("/feedback", json={"inference_id": INFERENCE_ID.upper(), "feedback": "agree"})
    assert response.status_code == 200
    assert response.json()["inference_id"] == INFERENCE_ID


# --- 404 ----------------------------------------------------------------------------------


def test_unknown_inference_answers_404_in_the_common_format(client, store):
    store.found = False
    response = client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": "agree"})
    assert response.status_code == 404
    body = response.json()
    assert body == {
        "detail": feedback.NOT_FOUND_DETAIL,
        "code": "NOT_FOUND",
        "errors": None,
    }
    assert "POST /predict" in body["detail"]


@pytest.mark.parametrize("value", ["agree", "disagree"])
def test_refusal_logs_do_not_contain_the_chosen_value(client, store, caplog, value):
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        store.found = False
        assert client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": value}).status_code == 404
        store.error = psycopg.OperationalError("panne")
        assert client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": value}).status_code == 503
    assert "agree" not in caplog.text
    assert "POST /feedback -> 404" in caplog.text
    assert "POST /feedback -> 503" in caplog.text


# --- 422 : choix fermé, sans texte libre -----------------------------------------------------


def test_free_text_is_refused_with_422_without_echoing_it(client, store, caplog):
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": FREE_TEXT})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_ERROR"
    assert body["errors"] == [
        {
            "field": "feedback",
            "message": "valeur non permise, valeurs possibles : 'agree' ou 'disagree'",
        }
    ]
    assert FREE_TEXT not in response.text
    assert FREE_TEXT not in caplog.text
    assert store.calls == []


@pytest.mark.parametrize(
    "value",
    [
        "",
        "Agree",
        "AGREE",
        " agree",
        "agree ",
        "oui",
        "neutral",
        None,
        1,
        0,
        True,
        ["agree"],
        {"text": "agree"},
    ],
)
def test_any_other_feedback_value_is_refused(client, store, value):
    response = client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": value})
    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_ERROR"
    assert [error["field"] for error in response.json()["errors"]] == ["feedback"]
    assert store.calls == []


def test_free_text_in_an_extra_field_is_refused(client, store):
    response = client.post(
        "/feedback",
        json={
            "inference_id": INFERENCE_ID,
            "feedback": "disagree",
            "comment": FREE_TEXT,
        },
    )
    assert response.status_code == 422
    assert response.json()["errors"] == [{"field": "comment", "message": "champ inconnu"}]
    assert FREE_TEXT not in response.text
    assert store.calls == []


@pytest.mark.parametrize("inference_id", ["abc", "", 123, None, INFERENCE_ID[:-1], FREE_TEXT])
def test_malformed_inference_id_is_refused(client, store, inference_id):
    response = client.post("/feedback", json={"inference_id": inference_id, "feedback": "agree"})
    assert response.status_code == 422
    assert [error["field"] for error in response.json()["errors"]] == ["inference_id"]
    assert FREE_TEXT not in response.text
    assert store.calls == []


@pytest.mark.parametrize(
    ("body", "fields"),
    [
        ({}, ["inference_id", "feedback"]),
        ({"feedback": "agree"}, ["inference_id"]),
        ({"inference_id": INFERENCE_ID}, ["feedback"]),
    ],
)
def test_missing_fields_are_listed(client, store, body, fields):
    response = client.post("/feedback", json=body)
    assert response.status_code == 422
    assert [error["field"] for error in response.json()["errors"]] == fields
    assert {error["message"] for error in response.json()["errors"]} == {"champ obligatoire manquant"}


@pytest.mark.parametrize("raw", ["not json", "[]", '"agree"', "null"])
def test_body_that_is_not_an_object_is_refused(client, store, raw):
    response = client.post("/feedback", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == "corps"
    assert store.calls == []


def test_get_is_not_allowed(client, store):
    response = client.get("/feedback")
    assert response.status_code == 405
    assert response.json()["code"] == "METHOD_NOT_ALLOWED"


# --- base indisponible ----------------------------------------------------------------------


def test_database_not_configured_answers_503(client):
    # conftest.py retire les variables POSTGRES_* : le vrai store_feedback lève DatabaseNotConfigured.
    response = client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": "agree"})
    assert response.status_code == 503
    assert response.json() == {
        "detail": feedback.DATABASE_UNAVAILABLE_DETAIL,
        "code": "DATABASE_UNAVAILABLE",
        "errors": None,
    }


@pytest.mark.parametrize(
    "error",
    [
        psycopg.OperationalError("hôte secret-7f3a"),
        psycopg.InterfaceError("secret-7f3a"),
    ],
)
def test_unreachable_database_answers_503_without_the_driver_message(client, store, caplog, error):
    store.error = error
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": "disagree"})
    assert response.status_code == 503
    assert response.json()["code"] == "DATABASE_UNAVAILABLE"
    assert "secret-7f3a" not in response.text
    assert "secret-7f3a" not in caplog.text
    assert type(error).__name__ in caplog.text


def test_other_database_error_answers_500_without_its_message(client, store, caplog):
    store.error = psycopg.errors.InsufficientPrivilege("droit refusé secret-7f3a")
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": "agree"})
    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "secret-7f3a" not in response.text + caplog.text


# --- store_feedback : complétion du schéma ---------------------------------------------------


class FakeConnection:
    def __init__(self):
        self.rollbacks = 0
        self.executed = []

    def execute(self, query, parameters=None):
        self.executed.append(query)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def fake_db(monkeypatch):
    """Faux connect(), save_feedback() et create_schema() pour store_feedback."""
    state = types.SimpleNamespace(connection=FakeConnection(), outcomes=[], saves=[], schemas=[])

    def save(inference_id, value, conn):
        state.saves.append((inference_id, value, conn))
        outcome = state.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(feedback, "connect", lambda: state.connection)
    monkeypatch.setattr(feedback, "save_feedback", save)
    monkeypatch.setattr(feedback, "create_schema", lambda conn: state.schemas.append(conn))
    return state


@pytest.mark.parametrize("missing", [psycopg.errors.UndefinedColumn, psycopg.errors.UndefinedTable])
def test_missing_column_or_table_is_created_then_written_once_more(fake_db, missing):
    fake_db.outcomes = [missing("absente"), True]
    assert feedback.store_feedback(UUID(INFERENCE_ID), "disagree") is True
    assert [(inference_id, value) for inference_id, value, _ in fake_db.saves] == [(UUID(INFERENCE_ID), "disagree")] * 2
    assert fake_db.connection.rollbacks == 1
    assert fake_db.schemas == [fake_db.connection]
    assert len(fake_db.saves) == 2
    assert all(conn is fake_db.connection for *_, conn in fake_db.saves)
    # L'attente de verrou est bornée dans chaque transaction, y compris après le rollback.
    assert fake_db.connection.executed == ["SET LOCAL lock_timeout = '2s'"] * 2


def test_schema_completion_is_tried_only_once(fake_db):
    fake_db.outcomes = [
        psycopg.errors.UndefinedColumn("absente"),
        psycopg.errors.UndefinedColumn("toujours"),
    ]
    with pytest.raises(psycopg.errors.UndefinedColumn):
        feedback.store_feedback(UUID(INFERENCE_ID), "agree")
    assert len(fake_db.schemas) == 1


def test_other_errors_do_not_touch_the_schema(fake_db):
    fake_db.outcomes = [psycopg.errors.InsufficientPrivilege("non")]
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        feedback.store_feedback(UUID(INFERENCE_ID), "agree")
    assert fake_db.schemas == []


@pytest.mark.parametrize("found", [True, False])
def test_first_write_is_kept_without_rollback_or_schema(fake_db, found):
    fake_db.outcomes = [found]
    assert feedback.store_feedback(UUID(INFERENCE_ID), "disagree") is found
    assert fake_db.connection.rollbacks == 0
    assert fake_db.schemas == []
    assert len(fake_db.saves) == 1
    assert fake_db.connection.executed == ["SET LOCAL lock_timeout = '2s'"]


def test_lock_wait_beyond_the_timeout_answers_503(client, store):
    store.error = psycopg.errors.LockNotAvailable("verrou secret-7f3a")
    response = client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": "agree"})
    assert response.status_code == 503
    assert response.json()["code"] == "DATABASE_UNAVAILABLE"
    assert "secret-7f3a" not in response.text


# --- save_feedback ---------------------------------------------------------------------------


class RecordingConnection:
    def __init__(self, rowcount):
        self.rowcount, self.executed = rowcount, []

    def execute(self, query, parameters):
        self.executed.append((query, parameters))
        return types.SimpleNamespace(rowcount=self.rowcount)


@pytest.mark.parametrize(("rowcount", "expected"), [(1, True), (0, False)])
def test_save_feedback_updates_only_a_successful_inference(rowcount, expected):
    conn = RecordingConnection(rowcount)
    assert inferences.save_feedback(UUID(INFERENCE_ID), "agree", conn) is expected
    ((query, parameters),) = conn.executed
    assert parameters == {"id": UUID(INFERENCE_ID), "feedback": "agree"}
    # Requête exacte : ni OR, ni RETURNING (qui demanderait SELECT sur feedback, hors des droits documentés).
    assert (
        " ".join(query.split())
        == "UPDATE inferences SET feedback = %(feedback)s WHERE id = %(id)s AND status = 'success'"
    )


@pytest.mark.parametrize(("rowcount", "expected"), [(1, True), (0, False)])
def test_save_feedback_opens_its_own_connection_without_one(monkeypatch, rowcount, expected):
    conn = RecordingConnection(rowcount)
    opened = []

    class Context:
        def __enter__(self):
            opened.append(conn)
            return conn

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(inferences, "connect", lambda: Context())
    assert inferences.save_feedback(UUID(INFERENCE_ID), "disagree") is expected
    assert opened == [conn]
    assert len(conn.executed) == 1


@pytest.mark.parametrize("value", ["", "Agree", "texte libre", None])
def test_save_feedback_refuses_other_values_before_the_database(value):
    conn = RecordingConnection(1)
    with pytest.raises(ValueError):
        inferences.save_feedback(UUID(INFERENCE_ID), value, conn)
    assert conn.executed == []


# --- contrat, schéma SQL et application ------------------------------------------------------


def test_openapi_announces_404_422_503_and_no_longer_501(client):
    responses = client.get("/openapi.json").json()["paths"]["/feedback"]["post"]["responses"]
    assert {"200", "404", "422", "500", "503"} <= set(responses)
    assert "501" not in responses


def test_schema_script_adds_the_column_idempotently_after_creating_the_table():
    script = init_db.SCHEMA_PATH.read_text(encoding="utf-8")
    alter = re.search(
        r"ALTER TABLE inferences ADD COLUMN IF NOT EXISTS feedback text\s+CONSTRAINT",
        script,
    )
    assert alter is not None
    assert script.index("CREATE TABLE IF NOT EXISTS inferences") < alter.start()
    assert "CHECK (feedback IN ('agree', 'disagree'))" in script
    # Un seul ALTER TABLE, et pas d'ADD CONSTRAINT séparé (qui échouerait au second passage).
    assert len(re.findall(r"^ALTER TABLE[^;]*;", script, re.MULTILINE)) == 1
    assert "ADD CONSTRAINT" not in script
    # L'attente du verrou exclusif de l'ALTER TABLE est bornée par create_schema(), pas par le script.
    assert "SET LOCAL" not in script


def _doc_section() -> str:
    text = DOC_PATH.read_text(encoding="utf-8")
    return text.split("## POST /feedback")[1].split("\n## ")[0]


def documented_grant(role: str) -> str:
    """GRANT donné par docs/api.md pour la route, appliqué au rôle demandé."""
    (grant,) = re.findall(r"`(GRANT [^`]+ TO <rôle>)`", _doc_section())
    return grant.replace("<rôle>", role)


def test_doc_lists_every_status_of_the_route(client):
    section = _doc_section()
    (errors_line,) = [line for line in section.splitlines() if line.startswith("Erreurs :")]
    responses = client.get("/openapi.json").json()["paths"]["/feedback"]["post"]["responses"]
    for status in set(responses) - {"200"}:
        assert status in errors_line, status
    # Une inférence en erreur répond 404 NOT_FOUND, annoncé sur la ligne des erreurs et dans les puces.
    assert "enregistrée en erreur" in errors_line
    assert "409" not in section  # pas de conflit : une inférence sans décision répond 404
    assert "elle répond 404 comme une inférence inconnue (`NOT_FOUND`)" in section


def test_doc_gives_the_lock_timeouts_of_the_code():
    assert feedback.FEEDBACK_LOCK_TIMEOUT == "2s"
    assert init_db.SCHEMA_LOCK_TIMEOUT == "5s"
    assert "attend au plus 2 s (`lock_timeout`)" in _doc_section()
    assert "borne à 5 s l'attente de chaque verrou" in DOC_PATH.read_text(encoding="utf-8")


def test_doc_grant_gives_exactly_the_rights_the_query_needs():
    assert documented_grant("r") == "GRANT SELECT (id, status), UPDATE (feedback) ON inferences TO r"


def test_route_is_wired_in_the_application(tmp_path):
    contract = {
        "version": "test-f1",
        "features": [{"name": "loan_amnt", "type": "numeric", "min": 0}],
    }
    contract_path, examples_path = (
        tmp_path / "features.yaml",
        tmp_path / "examples.json",
    )
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    examples_path.write_text(json.dumps({"moyen": {"loan_amnt": 1000}}), encoding="utf-8")
    client = TestClient(main.create_app(contract_path, examples_path, database_check=lambda: False))
    # Sans base configurée : la route est bien celle de F1 (503), plus la réponse vide 501.
    response = client.post("/feedback", json={"inference_id": INFERENCE_ID, "feedback": "agree"})
    assert response.status_code == 503
    assert response.json()["code"] == "DATABASE_UNAVAILABLE"

"""Tests de POST /feedback (tâche F1).

La première partie n'utilise aucune base : l'écriture est remplacée par un faux. La seconde
s'exécute contre un vrai PostgreSQL, comme tests/api/test_predict_postgres.py, et n'est lancée
que si les variables NEURAXION_TEST_POSTGRES_* sont renseignées (base jetable : la table
inferences y est supprimée puis recréée).
"""

import json
import logging
import os
import re
import secrets
import sys
import types
from uuid import UUID, uuid4

import numpy as np
import psycopg
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import feedback, main, served_model
from src.api.errors import install_error_handlers
from src.db import connection, inferences, init_db

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
def test_feedback_is_recorded_and_echoed(client, store, value):
    response = client.post(
        "/feedback", json={"inference_id": INFERENCE_ID, "feedback": value}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"inference_id": INFERENCE_ID, "feedback": value}
    assert store.calls == [(UUID(INFERENCE_ID), value)]


def test_uppercase_uuid_is_accepted_and_answered_in_canonical_form(client, store):
    response = client.post(
        "/feedback", json={"inference_id": INFERENCE_ID.upper(), "feedback": "agree"}
    )
    assert response.status_code == 200
    assert response.json()["inference_id"] == INFERENCE_ID


# --- 404 ----------------------------------------------------------------------------------


def test_unknown_inference_answers_404_in_the_common_format(client, store):
    store.found = False
    response = client.post(
        "/feedback", json={"inference_id": INFERENCE_ID, "feedback": "agree"}
    )
    assert response.status_code == 404
    body = response.json()
    assert body == {
        "detail": feedback.NOT_FOUND_DETAIL,
        "code": "NOT_FOUND",
        "errors": None,
    }
    assert "POST /predict" in body["detail"]


# --- 422 : choix fermé, sans texte libre -----------------------------------------------------


def test_free_text_is_refused_with_422_without_echoing_it(client, store, caplog):
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = client.post(
            "/feedback", json={"inference_id": INFERENCE_ID, "feedback": FREE_TEXT}
        )
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
    response = client.post(
        "/feedback", json={"inference_id": INFERENCE_ID, "feedback": value}
    )
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
    assert response.json()["errors"] == [
        {"field": "comment", "message": "champ inconnu"}
    ]
    assert FREE_TEXT not in response.text
    assert store.calls == []


@pytest.mark.parametrize(
    "inference_id", ["abc", "", 123, None, INFERENCE_ID[:-1], FREE_TEXT]
)
def test_malformed_inference_id_is_refused(client, store, inference_id):
    response = client.post(
        "/feedback", json={"inference_id": inference_id, "feedback": "agree"}
    )
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
    assert {error["message"] for error in response.json()["errors"]} == {
        "champ obligatoire manquant"
    }


@pytest.mark.parametrize("raw", ["not json", "[]", '"agree"', "null"])
def test_body_that_is_not_an_object_is_refused(client, store, raw):
    response = client.post(
        "/feedback", content=raw, headers={"content-type": "application/json"}
    )
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
    response = client.post(
        "/feedback", json={"inference_id": INFERENCE_ID, "feedback": "agree"}
    )
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
def test_unreachable_database_answers_503_without_the_driver_message(
    client, store, caplog, error
):
    store.error = error
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = client.post(
            "/feedback", json={"inference_id": INFERENCE_ID, "feedback": "disagree"}
        )
    assert response.status_code == 503
    assert response.json()["code"] == "DATABASE_UNAVAILABLE"
    assert "secret-7f3a" not in response.text
    assert "secret-7f3a" not in caplog.text
    assert type(error).__name__ in caplog.text


def test_other_database_error_answers_500_without_its_message(client, store, caplog):
    store.error = psycopg.errors.InsufficientPrivilege("droit refusé secret-7f3a")
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = client.post(
            "/feedback", json={"inference_id": INFERENCE_ID, "feedback": "agree"}
        )
    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "secret-7f3a" not in response.text + caplog.text


# --- store_feedback : complétion du schéma ---------------------------------------------------


class FakeConnection:
    def __init__(self):
        self.rollbacks = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def rollback(self):
        self.rollbacks += 1


@pytest.fixture
def fake_db(monkeypatch):
    """Faux connect(), save_feedback() et create_schema() pour store_feedback."""
    state = types.SimpleNamespace(
        connection=FakeConnection(), outcomes=[], saves=[], schemas=[]
    )

    def save(inference_id, value, conn):
        state.saves.append((inference_id, value, conn))
        outcome = state.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(feedback, "connect", lambda: state.connection)
    monkeypatch.setattr(feedback, "save_feedback", save)
    monkeypatch.setattr(
        feedback, "create_schema", lambda conn: state.schemas.append(conn)
    )
    return state


@pytest.mark.parametrize(
    "missing", [psycopg.errors.UndefinedColumn, psycopg.errors.UndefinedTable]
)
def test_missing_column_or_table_is_created_then_written_once_more(fake_db, missing):
    fake_db.outcomes = [missing("absente"), True]
    assert feedback.store_feedback(UUID(INFERENCE_ID), "agree") is True
    assert fake_db.connection.rollbacks == 1
    assert fake_db.schemas == [fake_db.connection]
    assert len(fake_db.saves) == 2
    assert all(conn is fake_db.connection for *_, conn in fake_db.saves)


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


def test_unknown_inference_is_reported_as_false(fake_db):
    fake_db.outcomes = [False]
    assert feedback.store_feedback(UUID(INFERENCE_ID), "disagree") is False
    assert fake_db.schemas == []


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
    assert "UPDATE inferences SET feedback" in query
    assert "status = 'success'" in query


@pytest.mark.parametrize("value", ["", "Agree", "texte libre", None])
def test_save_feedback_refuses_other_values_before_the_database(value):
    conn = RecordingConnection(1)
    with pytest.raises(ValueError):
        inferences.save_feedback(UUID(INFERENCE_ID), value, conn)
    assert conn.executed == []


# --- contrat, schéma SQL et application ------------------------------------------------------


def test_openapi_announces_404_422_503_and_no_longer_501(client):
    responses = client.get("/openapi.json").json()["paths"]["/feedback"]["post"][
        "responses"
    ]
    assert {"200", "404", "422", "503"} <= set(responses)
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
    examples_path.write_text(
        json.dumps({"moyen": {"loan_amnt": 1000}}), encoding="utf-8"
    )
    client = TestClient(
        main.create_app(contract_path, examples_path, database_check=lambda: False)
    )
    # Sans base configurée : la route est bien celle de F1 (503), plus la réponse vide 501.
    response = client.post(
        "/feedback", json={"inference_id": INFERENCE_ID, "feedback": "agree"}
    )
    assert response.status_code == 503
    assert response.json()["code"] == "DATABASE_UNAVAILABLE"


# === Contre un vrai PostgreSQL ===============================================================

PREFIX = "NEURAXION_TEST_POSTGRES_"
ENVIRON = {
    f"POSTGRES_{key}": os.environ[PREFIX + key]
    for key in ("HOST", "PORT", "USER", "PASSWORD", "DB")
    if os.environ.get(PREFIX + key)
}
postgres = pytest.mark.skipif(
    not {"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"} <= set(ENVIRON),
    reason="PostgreSQL de test non configuré (variables NEURAXION_TEST_POSTGRES_*).",
)

CONTRACT = {
    "version": "test-f1-pg",
    "features": [
        {"name": "loan_amnt", "type": "numeric", "min": 1000, "max": 40000},
        {
            "name": "home_ownership",
            "type": "categorical",
            "allowed": ["RENT", "OWN", "MORTGAGE"],
        },
    ],
}
FEATURES = [item["name"] for item in CONTRACT["features"]]
VALID = {"loan_amnt": 12000, "home_ownership": "RENT"}

# Table telle que la créait src/db/schema.sql avant F1 (sans la colonne feedback).
TABLE_BEFORE_F1 = init_db.SCHEMA_PATH.read_text(encoding="utf-8").split("-- Tâche F1")[
    0
]


class FakeModel:
    classes_ = np.array([0, 1])

    def predict_proba(self, frame):
        return np.array([[0.58, 0.42]])


def install_champion(monkeypatch, found):
    module = types.ModuleType(served_model.CHAMPION_MODULE)
    module.load_champion = lambda: found
    monkeypatch.setitem(sys.modules, served_model.CHAMPION_MODULE, module)


def champion():
    return {
        "model": FakeModel(),
        "version": "1",
        "decision_threshold": 0.4,
        "features": FEATURES,
    }


def feedback_of(inference_id) -> str | None:
    with connection.connect(ENVIRON) as conn:
        (value,) = conn.execute(
            "SELECT feedback FROM inferences WHERE id = %s", (inference_id,)
        ).fetchone()
    return value


def column_names() -> list[str]:
    with connection.connect(ENVIRON) as conn:
        found = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'inferences' ORDER BY ordinal_position"
        ).fetchall()
    return [name for (name,) in found]


@pytest.fixture
def pg(monkeypatch, tmp_path):
    """Table vide, variables POSTGRES_* posées et fabrique d'application complète."""
    with connection.connect(ENVIRON) as conn:
        conn.execute("DROP TABLE IF EXISTS inferences")
    for key, value in ENVIRON.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
    install_champion(monkeypatch, champion())

    def make_app():
        contract_path, examples_path = (
            tmp_path / "features.yaml",
            tmp_path / "examples.json",
        )
        contract_path.write_text(yaml.safe_dump(CONTRACT), encoding="utf-8")
        examples_path.write_text(json.dumps({"moyen": VALID}), encoding="utf-8")
        return main.create_app(contract_path, examples_path)

    return make_app


@postgres
def test_pg_create_schema_adds_the_column_and_its_check_idempotently(pg):
    for _ in range(3):
        init_db.create_schema()
    assert column_names()[-1] == "feedback"
    assert column_names().count("feedback") == 1
    with connection.connect(ENVIRON) as conn:
        checks = conn.execute(
            "SELECT conname FROM pg_constraint WHERE conrelid = 'inferences'::regclass AND conname = 'inferences_feedback_values'"
        ).fetchall()
    assert len(checks) == 1


@postgres
def test_pg_table_created_before_f1_is_completed_and_keeps_its_rows(pg):
    with connection.connect(ENVIRON) as conn:
        conn.execute(TABLE_BEFORE_F1)
        conn.execute(
            "INSERT INTO inferences (input_data, status, error_message, latency_ms) "
            "VALUES ('{}'::jsonb, 'error', 'ancienne ligne', 1)"
        )
    assert "feedback" not in column_names()
    init_db.create_schema()
    assert "feedback" in column_names()
    with connection.connect(ENVIRON) as conn:
        assert conn.execute(
            "SELECT count(*), count(feedback) FROM inferences"
        ).fetchone() == (1, 0)


@postgres
def test_pg_feedback_after_a_prediction_is_stored_then_replaced(pg):
    with TestClient(pg()) as client:
        inference_id = client.post("/predict", json=VALID).json()["inference_id"]
        assert feedback_of(inference_id) is None
        first = client.post(
            "/feedback", json={"inference_id": inference_id, "feedback": "agree"}
        )
        assert first.status_code == 200, first.text
        assert first.json() == {"inference_id": inference_id, "feedback": "agree"}
        assert feedback_of(inference_id) == "agree"
        second = client.post(
            "/feedback", json={"inference_id": inference_id, "feedback": "disagree"}
        )
        assert second.status_code == 200
    assert feedback_of(inference_id) == "disagree"


@postgres
def test_pg_unknown_inference_answers_404_and_changes_nothing(pg):
    with TestClient(pg()) as client:
        inference_id = client.post("/predict", json=VALID).json()["inference_id"]
        response = client.post(
            "/feedback", json={"inference_id": str(uuid4()), "feedback": "agree"}
        )
    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"
    assert feedback_of(inference_id) is None


@postgres
def test_pg_inference_in_error_has_no_decision_and_answers_404(pg, monkeypatch):
    install_champion(monkeypatch, None)
    with TestClient(pg()) as client:
        assert client.post("/predict", json=VALID).status_code == 503
        with connection.connect(ENVIRON) as conn:
            (error_id,) = conn.execute(
                "SELECT id FROM inferences WHERE status = 'error'"
            ).fetchone()
        response = client.post(
            "/feedback", json={"inference_id": str(error_id), "feedback": "disagree"}
        )
    assert response.status_code == 404
    assert feedback_of(error_id) is None


@postgres
def test_pg_free_text_is_refused_by_the_api_and_by_the_database(pg):
    with TestClient(pg()) as client:
        inference_id = client.post("/predict", json=VALID).json()["inference_id"]
        response = client.post(
            "/feedback", json={"inference_id": inference_id, "feedback": FREE_TEXT}
        )
    assert response.status_code == 422
    assert feedback_of(inference_id) is None
    with (
        connection.connect(ENVIRON) as conn,
        pytest.raises(psycopg.errors.CheckViolation),
    ):
        conn.execute(
            "UPDATE inferences SET feedback = %s WHERE id = %s",
            (FREE_TEXT, inference_id),
        )


@postgres
def test_pg_column_dropped_after_startup_is_added_back_by_the_route(pg):
    with TestClient(pg()) as client:
        inference_id = client.post("/predict", json=VALID).json()["inference_id"]
        with connection.connect(ENVIRON) as conn:
            conn.execute("ALTER TABLE inferences DROP COLUMN feedback")
        response = client.post(
            "/feedback", json={"inference_id": inference_id, "feedback": "agree"}
        )
    assert response.status_code == 200, response.text
    assert feedback_of(inference_id) == "agree"


@postgres
def test_pg_database_absent_at_startup_then_back(pg, monkeypatch):
    monkeypatch.setenv(
        "POSTGRES_PORT", "1"
    )  # port fermé au démarrage : ni table ni colonne
    with TestClient(pg()) as client:
        down = client.post(
            "/feedback", json={"inference_id": str(uuid4()), "feedback": "agree"}
        )
        monkeypatch.setenv("POSTGRES_PORT", ENVIRON.get("POSTGRES_PORT", "5432"))
        # La table n'existe pas encore : la route la crée, puis répond 404 pour un id inconnu.
        missing = client.post(
            "/feedback", json={"inference_id": str(uuid4()), "feedback": "agree"}
        )
    assert down.status_code == 503
    assert down.json()["code"] == "DATABASE_UNAVAILABLE"
    assert missing.status_code == 404
    assert "feedback" in column_names()


@postgres
def test_pg_role_with_only_the_documented_rights_can_record_feedback(pg, monkeypatch):
    with TestClient(pg()) as client:
        inference_id = client.post("/predict", json=VALID).json()["inference_id"]
    role, password = "neuraxion_feedback_limitee", secrets.token_urlsafe(16)
    with connection.connect(ENVIRON) as conn:
        conn.execute(f"DROP ROLE IF EXISTS {role}")
        conn.execute(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'")
        conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        conn.execute(
            f"GRANT SELECT (id, status), UPDATE (feedback) ON inferences TO {role}"
        )
    try:
        monkeypatch.setenv("POSTGRES_USER", role)
        monkeypatch.setenv("POSTGRES_PASSWORD", password)
        with TestClient(pg()) as client:
            response = client.post(
                "/feedback", json={"inference_id": inference_id, "feedback": "disagree"}
            )
            unknown = client.post(
                "/feedback", json={"inference_id": str(uuid4()), "feedback": "agree"}
            )
        assert response.status_code == 200, response.text
        assert unknown.status_code == 404
        assert feedback_of(inference_id) == "disagree"
    finally:
        with connection.connect(ENVIRON) as conn:
            conn.execute(f"DROP OWNED BY {role}")
            conn.execute(f"DROP ROLE {role}")

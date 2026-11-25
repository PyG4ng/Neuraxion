"""Tests de A2 contre un vrai PostgreSQL : création du schéma et enregistrement des inférences.

Base, schéma temporaire et faux @champion : voir conftest.py. Sautés sans POSTGRES_*.
"""

import os
import secrets
import subprocess
import sys
import threading
from datetime import UTC, datetime
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql

from src.api import main
from src.db import inferences, init_db

CONTRACT_FIELDS = {"loan_amnt", "annual_inc", "home_ownership", "dti", "purpose"}


def test_create_schema_is_idempotent_and_builds_the_documented_table(database):
    init_db.create_schema()
    init_db.create_schema()
    with database.connect() as conn:
        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = %s AND table_name = 'inferences'",
            (database.schema,),
        ).fetchall()
        indexes = conn.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = %s AND tablename = 'inferences'", (database.schema,)
        ).fetchall()
    assert len(columns) == 12  # 11 colonnes de A2, plus feedback (F1)
    assert {"inferences_created_at_idx", "inferences_model_version_idx", "inferences_pkey"} <= {i for (i,) in indexes}


def test_concurrent_create_schema_calls_do_not_fail():
    failures = []

    def run():
        try:
            init_db.create_schema()
        except Exception as error:  # noqa: BLE001
            failures.append(type(error).__name__)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_init_db_module_command_creates_the_table(database):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # POSTGRES_* et PGOPTIONS posés par conftest.py
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, "-m", "src.db.init_db"],
            cwd=main.ROOT,
            capture_output=True,
            encoding="utf-8",
            env=env,
            timeout=60,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "Table inferences prête" in result.stdout
        assert database.environ["POSTGRES_PASSWORD"] not in result.stdout + result.stderr
    assert database.rows() == []


def test_prediction_creates_a_row_with_the_version(make_app, registry, database, valid_request):
    registry.serve(version="1", threshold=0.4)
    with TestClient(make_app()) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/predict", json={**valid_request, "dti": 18.4})
    assert response.status_code == 200, response.text
    body = response.json()
    (row,) = database.rows()
    assert str(row["id"]) == body["inference_id"]
    assert row["created_at"] == datetime.fromisoformat(body["created_at"])
    assert row["status"] == "success"
    assert (row["model_name"], row["model_version"]) == ("credit-default", "1")
    assert row["prediction"] == pytest.approx(0.42)
    assert row["decision"] == "reject" == body["decision"]
    assert row["decision_threshold"] == 0.4
    assert row["error_message"] is None
    assert row["latency_ms"] >= 0
    assert row["input_data"] == {
        "loan_amnt": 12000.0,
        "annual_inc": 55000.0,
        "home_ownership": "RENT",
        "dti": 18.4,
        "purpose": None,
    }


def test_the_row_records_the_name_of_the_served_model(make_app, registry, database, valid_request, monkeypatch):
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "credit-default-v2")
    registry.serve(version="6")
    with TestClient(make_app()) as client:
        body = client.post("/predict", json=valid_request).json()
    (row,) = database.rows()
    assert (row["model_name"], row["model_version"]) == (body["model_name"], body["model_version"])
    assert (row["model_name"], row["model_version"]) == ("credit-default-v2", "6")


def test_optional_field_absent_is_stored_as_null(make_app, registry, database, valid_request):
    registry.serve()
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=valid_request).status_code == 200
    (row,) = database.rows()
    assert set(row["input_data"]) == CONTRACT_FIELDS
    assert row["input_data"]["dti"] is None


def test_without_champion_503_and_an_error_row(make_app, database, valid_request):
    with TestClient(make_app()) as client:
        response = client.post("/predict", json=valid_request)
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"
    (row,) = database.rows()
    assert row["status"] == "error"
    assert row["error_message"] == "Aucun modèle validé n'est disponible."
    assert row["model_version"] is None


def test_model_failure_gives_an_error_row_with_the_version(make_app, registry, database, valid_request):
    registry.serve(version="2", error=ValueError("boum"))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=valid_request).status_code == 500
    (row,) = database.rows()
    assert (row["status"], row["model_version"]) == ("error", "2")
    assert row["error_message"] == "Le modèle a échoué pendant la prédiction (ValueError)."


def test_validation_error_creates_no_row(make_app, registry, database, valid_request):
    registry.serve()
    with TestClient(make_app()) as client:
        assert client.post("/predict", json={**valid_request, "pseudo": "x"}).status_code == 422
    assert database.rows() == []


def test_table_dropped_after_startup_is_recreated(make_app, registry, database, valid_request):
    registry.serve()
    with TestClient(make_app()) as client:
        with database.connect() as conn:
            conn.execute(sql.SQL("DROP TABLE {}.inferences").format(sql.Identifier(database.schema)))
        assert client.post("/predict", json=valid_request).status_code == 200
    assert len(database.rows()) == 1


def test_check_constraints_refuse_inconsistent_rows(database):
    init_db.create_schema()
    record = inferences.InferenceRecord(
        id=uuid4(),
        created_at=datetime.now(UTC),
        input_data={"loan_amnt": 1.0},
        status="success",
        latency_ms=1,
        prediction=0.9,
        decision="approve",  # incohérent avec le seuil
        decision_threshold=0.5,
        model_name="credit-default",
        model_version="1",
    )
    with pytest.raises(psycopg.errors.CheckViolation):
        inferences.save_inference(record)
    assert database.rows() == []


def test_save_inference_without_a_connection_commits_the_row(database):
    init_db.create_schema()
    record = inferences.InferenceRecord(
        id=uuid4(),
        created_at=datetime.now(UTC),
        input_data={"loan_amnt": 1.0},
        status="error",
        latency_ms=1,
        error_message="Ligne de test.",
    )
    assert inferences.save_inference(record) == record.id
    assert [row["id"] for row in database.rows()] == [record.id]  # lue par une autre connexion


def test_table_is_created_by_the_first_inference_when_the_database_was_down_at_startup(
    make_app, registry, database, valid_request, monkeypatch
):
    registry.serve()
    monkeypatch.setenv("POSTGRES_PORT", "1")  # port fermé au démarrage
    with TestClient(make_app()) as client:
        monkeypatch.setenv("POSTGRES_PORT", database.environ.get("POSTGRES_PORT", "5432"))
        statuses = [client.post("/predict", json=valid_request).status_code for _ in range(3)]
    assert statuses == [200, 200, 200]
    assert len(database.rows()) == 3


def test_nul_character_is_refused_before_the_database(make_app, registry, database, valid_request):
    registry.serve()
    with TestClient(make_app()) as client:
        response = client.post("/predict", json={**valid_request, "purpose": "a" + chr(0) + "b"})
    assert response.status_code == 422
    assert database.rows() == []


def test_role_allowed_only_to_insert_can_record_inferences(make_app, registry, database, valid_request, monkeypatch):
    init_db.create_schema()
    role, password = f"neuraxion_api_limitee_{secrets.token_hex(4)}", secrets.token_urlsafe(16)
    schema = sql.Identifier(database.schema)
    with database.connect() as conn:
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
    try:
        with database.connect() as conn:
            # Droits limités au schéma de test : le schéma public de la base n'est pas touché.
            conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(schema, sql.Identifier(role)))
            conn.execute(  # ni SELECT, ni CREATE
                sql.SQL("GRANT INSERT ON {}.inferences TO {}").format(schema, sql.Identifier(role))
            )
            rights = conn.execute(
                "SELECT has_table_privilege(%s, 'inferences', 'INSERT'), has_table_privilege(%s, 'inferences', 'SELECT'),"
                " has_schema_privilege(%s, %s, 'CREATE')",
                (role, role, role, database.schema),
            ).fetchone()
        assert rights == (True, False, False)
        monkeypatch.setenv("POSTGRES_USER", role)
        monkeypatch.setenv("POSTGRES_PASSWORD", password)
        registry.serve()
        with TestClient(make_app()) as client:
            statuses = [client.post("/predict", json=valid_request).status_code for _ in range(2)]
        assert statuses == [200, 200]
        assert len(database.rows()) == 2
    finally:
        with database.connect() as conn:
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))

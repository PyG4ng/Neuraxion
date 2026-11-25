"""Tests de A2 contre un vrai PostgreSQL. Ignorés sans configuration.

Ils lisent NEURAXION_TEST_POSTGRES_HOST, _PORT, _USER, _PASSWORD et _DB (et non POSTGRES_*,
que conftest.py retire pour les autres tests). Base jetable : la table inferences y est
supprimée puis recréée. Exemple avec Docker :

    docker run -d --rm --name neuraxion-test-pg -e POSTGRES_PASSWORD=<aléatoire> -p 127.0.0.1:55432:5432 postgres:17-alpine
"""

import json
import os
import secrets
import subprocess
import sys
import threading
import types
from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import psycopg
import pytest
import yaml
from fastapi.testclient import TestClient

from src.api import main, served_model
from src.db import connection, inferences, init_db

PREFIX = "NEURAXION_TEST_POSTGRES_"
ENVIRON = {
    f"POSTGRES_{key}": os.environ[PREFIX + key]
    for key in ("HOST", "PORT", "USER", "PASSWORD", "DB")
    if os.environ.get(PREFIX + key)
}
pytestmark = pytest.mark.skipif(
    not {"POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"} <= set(ENVIRON),
    reason="PostgreSQL de test non configuré (variables NEURAXION_TEST_POSTGRES_*).",
)

CONTRACT = {
    "version": "test-a2-pg",
    "features": [
        {"name": "loan_amnt", "type": "numeric", "min": 1000, "max": 40000},
        {"name": "annual_inc", "type": "numeric", "min": 0},
        {"name": "home_ownership", "type": "categorical", "allowed": ["RENT", "OWN", "MORTGAGE"]},
        {"name": "dti", "type": "numeric", "required": False},
        {"name": "purpose", "type": "categorical", "required": False},
    ],
}
FEATURES = [feature["name"] for feature in CONTRACT["features"]]
VALID = {"loan_amnt": 12000, "annual_inc": 55000, "home_ownership": "RENT"}


class FakeModel:
    classes_ = np.array([0, 1])

    def __init__(self, probability=0.42, error=None):
        self.probability, self.error = probability, error

    def predict_proba(self, frame):
        if self.error:
            raise self.error
        return np.array([[1 - self.probability, self.probability]])


def install_champion(monkeypatch, found):
    module = types.ModuleType(served_model.CHAMPION_MODULE)
    module.load_champion = lambda: found
    monkeypatch.setitem(sys.modules, served_model.CHAMPION_MODULE, module)


def champion(model=None, version="1", threshold=0.4):
    return {"model": model or FakeModel(), "version": version, "decision_threshold": threshold, "features": FEATURES}


def rows(where: str = "TRUE") -> list[dict]:
    with connection.connect(ENVIRON) as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cursor:
        return cursor.execute(f"SELECT * FROM inferences WHERE {where} ORDER BY created_at").fetchall()


@pytest.fixture(autouse=True)
def fresh_table(monkeypatch):
    with connection.connect(ENVIRON) as conn:
        conn.execute("DROP TABLE IF EXISTS inferences")
    for key, value in ENVIRON.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)


@pytest.fixture
def make_app(tmp_path):
    def factory():
        contract_path, examples_path = tmp_path / "features.yaml", tmp_path / "examples.json"
        contract_path.write_text(yaml.safe_dump(CONTRACT), encoding="utf-8")
        examples_path.write_text(json.dumps({"moyen": VALID}), encoding="utf-8")
        return main.create_app(contract_path, examples_path)

    return factory


def test_create_schema_is_idempotent_and_builds_the_documented_table():
    init_db.create_schema()
    init_db.create_schema()
    with connection.connect(ENVIRON) as conn:
        columns = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'inferences'"
        ).fetchall()
        indexes = conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'inferences'").fetchall()
    assert len(columns) == 11
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


def test_init_db_module_command_creates_the_table():
    env = {key: value for key, value in os.environ.items() if not key.startswith("POSTGRES_")} | ENVIRON
    env["PYTHONIOENCODING"] = "utf-8"
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
        assert ENVIRON["POSTGRES_PASSWORD"] not in result.stdout + result.stderr


def test_prediction_creates_a_row_with_the_version(make_app, monkeypatch):
    install_champion(monkeypatch, champion(version="1", threshold=0.4))
    with TestClient(make_app()) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/predict", json={**VALID, "dti": 18.4})
    assert response.status_code == 200, response.text
    body = response.json()
    (row,) = rows()
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


def test_optional_field_absent_is_stored_as_null(make_app, monkeypatch):
    install_champion(monkeypatch, champion())
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).status_code == 200
    (row,) = rows()
    assert set(row["input_data"]) == set(FEATURES)
    assert row["input_data"]["dti"] is None


def test_without_champion_503_and_an_error_row(make_app, monkeypatch):
    install_champion(monkeypatch, None)
    with TestClient(make_app()) as client:
        response = client.post("/predict", json=VALID)
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"
    (row,) = rows()
    assert row["status"] == "error"
    assert row["error_message"] == "Aucun modèle validé n'est disponible."
    assert row["model_version"] is None


def test_model_failure_gives_an_error_row_with_the_version(make_app, monkeypatch):
    install_champion(monkeypatch, champion(FakeModel(error=ValueError("boum")), version="2"))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).status_code == 500
    (row,) = rows()
    assert (row["status"], row["model_version"]) == ("error", "2")
    assert row["error_message"] == "Le modèle a échoué pendant la prédiction (ValueError)."


def test_validation_error_creates_no_row(make_app, monkeypatch):
    install_champion(monkeypatch, champion())
    with TestClient(make_app()) as client:
        assert client.post("/predict", json={**VALID, "pseudo": "x"}).status_code == 422
    assert rows() == []


def test_table_dropped_after_startup_is_recreated(make_app, monkeypatch):
    install_champion(monkeypatch, champion())
    with TestClient(make_app()) as client:
        with connection.connect(ENVIRON) as conn:
            conn.execute("DROP TABLE inferences")
        assert client.post("/predict", json=VALID).status_code == 200
    assert len(rows()) == 1


def test_check_constraints_refuse_inconsistent_rows():
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
    assert rows() == []


def test_table_is_created_by_the_first_inference_when_the_database_was_down_at_startup(make_app, monkeypatch):
    install_champion(monkeypatch, champion())
    monkeypatch.setenv("POSTGRES_PORT", "1")  # port fermé au démarrage
    with TestClient(make_app()) as client:
        monkeypatch.setenv("POSTGRES_PORT", ENVIRON.get("POSTGRES_PORT", "5432"))
        statuses = [client.post("/predict", json=VALID).status_code for _ in range(3)]
    assert statuses == [200, 200, 200]
    assert len(rows()) == 3


def test_nul_character_is_refused_before_the_database(make_app, monkeypatch):
    install_champion(monkeypatch, champion())
    with TestClient(make_app()) as client:
        response = client.post("/predict", json={**VALID, "purpose": "a" + chr(0) + "b"})
    assert response.status_code == 422
    assert rows() == []


def test_role_allowed_only_to_insert_can_record_inferences(make_app, monkeypatch):
    init_db.create_schema()
    role, password = "neuraxion_api_limitee", secrets.token_urlsafe(16)
    with connection.connect(ENVIRON) as conn:
        conn.execute(f"DROP ROLE IF EXISTS {role}")
        conn.execute(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'")
        conn.execute(f"REVOKE CREATE ON SCHEMA public FROM {role}")
        conn.execute("REVOKE CREATE ON SCHEMA public FROM PUBLIC")
        conn.execute(f"GRANT INSERT ON inferences TO {role}")  # ni SELECT, ni CREATE
        rights = conn.execute(
            "SELECT has_table_privilege(%s, 'inferences', 'INSERT'), has_table_privilege(%s, 'inferences', 'SELECT')",
            (role, role),
        ).fetchone()
    assert rights == (True, False)
    try:
        monkeypatch.setenv("POSTGRES_USER", role)
        monkeypatch.setenv("POSTGRES_PASSWORD", password)
        install_champion(monkeypatch, champion())
        with TestClient(make_app()) as client:
            statuses = [client.post("/predict", json=VALID).status_code for _ in range(2)]
        assert statuses == [200, 200]
        assert len(rows()) == 2
    finally:
        with connection.connect(ENVIRON) as conn:
            conn.execute(f"DROP OWNED BY {role}")
            conn.execute(f"DROP ROLE {role}")

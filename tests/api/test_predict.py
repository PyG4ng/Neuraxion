"""Tests de POST /predict et de l'enregistrement des inférences (tâche A2).

Aucun test de ce fichier ne demande PostgreSQL ni MLflow : la base est remplacée par une
fausse connexion qui garde les requêtes SQL reçues, et le @champion par un faux module
src.registry.champion (l'interface de R1, issue #8). Les tests contre un vrai PostgreSQL
sont dans tests/db/test_predict_postgres.py.
"""

import json
import logging
import math
import re
import sys
import types
from datetime import datetime
from decimal import Decimal
from uuid import UUID

import numpy as np
import psycopg
import pytest
import yaml
from fastapi.testclient import TestClient

from src.api import main, predict, served_model
from src.api.schemas import ErrorResponse, PredictResponse, build_request_model
from src.db import inferences, init_db
from src.db.connection import DatabaseNotConfigured

CONTRACT = {
    "version": "test-a2",
    "features": [
        {"name": "loan_amnt", "type": "numeric", "min": 1000, "max": 40000},
        {"name": "annual_inc", "type": "numeric", "min": 0},
        {"name": "open_acc", "type": "integer", "min": 0, "required": False},
        {"name": "home_ownership", "type": "categorical", "allowed": ["RENT", "OWN", "MORTGAGE"]},
        {"name": "purpose", "type": "categorical", "required": False},
    ],
}
FEATURES = [feature["name"] for feature in CONTRACT["features"]]
EXAMPLES = {"moyen": {"loan_amnt": 12000, "annual_inc": 55000, "home_ownership": "RENT"}}
VALID = {"loan_amnt": 12000, "annual_inc": 55000, "open_acc": 4, "home_ownership": "RENT"}
SECRET = "Jean-Dupont-0612345678"
REPO_ROOT = main.ROOT
DOC_TEXT = (REPO_ROOT / "docs" / "api.md").read_text(encoding="utf-8")


# --- doublures ---------------------------------------------------------------


class FakeModel:
    """Modèle factice : probabilité fixe, ou calculée à partir de la ligne reçue."""

    classes_ = np.array([0, 1])

    def __init__(self, probability=0.2, error=None):
        self.probability = probability
        self.error = error
        self.frames = []

    def predict_proba(self, frame):
        self.frames.append(frame)
        if self.error is not None:
            raise self.error
        p = self.probability(frame) if callable(self.probability) else self.probability
        return np.array([[1 - p, p]]) if np.isscalar(p) else p


def champion(model=None, version="3", threshold=0.31, features=None, name="credit-default", **extra):
    found = {
        "model": model or FakeModel(),
        "name": name,
        "version": version,
        "decision_threshold": threshold,
        "features": list(FEATURES) if features is None else features,
    }
    found.update(extra)
    return found


class Registry:
    """Faux module src.registry.champion : load_champion() renvoie ce qui est posé dans current."""

    def __init__(self, monkeypatch, current=None):
        self.current = current
        self.calls = 0
        module = types.ModuleType(served_model.CHAMPION_MODULE)
        module.load_champion = self.load_champion
        monkeypatch.setitem(sys.modules, served_model.CHAMPION_MODULE, module)

    def load_champion(self):
        self.calls += 1
        if isinstance(self.current, BaseException):
            raise self.current
        return self.current() if callable(self.current) else self.current


class FakeConnection:
    def __init__(self, database):
        self.database = database

    def __enter__(self):
        return self

    def __exit__(self, kind, error, trace):
        if error is None:
            self.database.commits += 1
        return False

    def transaction(self):
        return self

    def rollback(self):
        self.database.rollbacks += 1

    def execute(self, query, params=None):
        self.database.queries.append(query)
        if query is inferences.INSERT_INFERENCE:
            if self.database.insert_error is not None:
                error, self.database.insert_error = self.database.insert_error, None
                raise error
            if not self.database.table_exists:
                raise psycopg.errors.UndefinedTable("relation absente")
            row = dict(params)
            row["input_data"] = row["input_data"].obj
            self.database.rows.append(row)
        elif "CREATE TABLE" in query:
            if self.database.create_error is not None:
                raise self.database.create_error
            self.database.schema_runs += 1
            self.database.table_exists = True


class FakeDatabase:
    def __init__(self):
        self.up = True
        self.rows = []
        self.queries = []
        self.schema_runs = 0
        self.commits = 0
        self.rollbacks = 0
        self.insert_error = None
        self.create_error = None
        self.table_exists = False

    def connect(self, environ=None):
        if self.up is not True:
            raise self.up if isinstance(self.up, BaseException) else psycopg.OperationalError("refusé")
        return FakeConnection(self)


@pytest.fixture
def database(monkeypatch):
    fake = FakeDatabase()
    monkeypatch.setattr(predict, "connect", fake.connect)
    return fake


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
    return Registry(monkeypatch, champion())


def write_files(tmp_path, contract=CONTRACT, examples=EXAMPLES):
    contract_path = tmp_path / "features.yaml"
    examples_path = tmp_path / "examples.json"
    contract_path.write_text(yaml.safe_dump(contract, allow_unicode=True), encoding="utf-8")
    examples_path.write_text(json.dumps(examples), encoding="utf-8")
    return contract_path, examples_path


@pytest.fixture
def make_app(tmp_path):
    def factory(contract=CONTRACT, examples=EXAMPLES):
        return main.create_app(*write_files(tmp_path, contract, examples), database_check=lambda: True)

    return factory


@pytest.fixture
def client(make_app, database, registry):
    with TestClient(make_app()) as test_client:
        yield test_client


def doc_error_example() -> dict:
    found = __import__("re").findall(r"<!-- exemple: ErrorResponse -->\s*```json\s*(.*?)```", DOC_TEXT, 16)
    return [json.loads(body) for body in found if "MODEL_UNAVAILABLE" in body][0]


# --- prédiction réussie --------------------------------------------------------


def test_prediction_returns_a_decision_and_records_one_row_with_the_version(client, database):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = PredictResponse.model_validate(response.json())
    assert body.default_probability == pytest.approx(0.2)
    assert body.decision == "approve"
    assert (body.model_name, body.model_version, body.decision_threshold) == ("credit-default", "3", 0.31)
    (row,) = database.rows
    assert row["id"] == body.inference_id
    assert row["created_at"] == body.created_at
    assert row["status"] == "success"
    assert row["model_version"] == "3"
    assert row["model_name"] == "credit-default"
    assert row["prediction"] == pytest.approx(0.2)
    assert row["decision"] == "approve"
    assert row["decision_threshold"] == 0.31
    assert row["error_message"] is None
    assert isinstance(row["latency_ms"], int) and row["latency_ms"] >= 0


def test_only_contract_fields_are_stored_in_input_data(client, database):
    assert client.post("/predict", json=VALID).status_code == 200
    stored = database.rows[0]["input_data"]
    assert set(stored) == set(FEATURES)
    assert stored == {**VALID, "purpose": None}


def test_unknown_fields_are_refused_and_not_recorded(client, database):
    response = client.post("/predict", json={**VALID, "nickname": SECRET})
    assert response.status_code == 422
    assert database.rows == []


def test_validation_errors_are_not_recorded(client, database):
    assert client.post("/predict", json={**VALID, "loan_amnt": -5}).status_code == 422
    assert client.post("/predict", content="{", headers={"content-type": "application/json"}).status_code == 422
    assert database.rows == []


def test_created_at_has_a_timezone_and_the_id_is_a_uuid(client):
    body = client.post("/predict", json=VALID).json()
    assert datetime.fromisoformat(body["created_at"]).tzinfo is not None
    UUID(body["inference_id"])


def test_each_prediction_gets_its_own_id(client, database):
    ids = {client.post("/predict", json=VALID).json()["inference_id"] for _ in range(3)}
    assert len(ids) == 3
    assert len(database.rows) == 3


@pytest.mark.parametrize(
    ("probability", "decision"),
    [(0.31, "reject"), (math.nextafter(0.31, 0), "approve"), (0.0, "approve"), (1.0, "reject"), (0.9, "reject")],
)
def test_decision_follows_the_model_threshold(make_app, database, monkeypatch, probability, decision):
    Registry(monkeypatch, champion(FakeModel(probability), threshold=0.31))
    with TestClient(make_app()) as client:
        body = client.post("/predict", json=VALID).json()
    assert body["decision"] == decision
    assert database.rows[0]["decision"] == decision


def test_threshold_zero_rejects_everything(make_app, database, monkeypatch):
    Registry(monkeypatch, champion(FakeModel(0.0), threshold=0.0))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).json()["decision"] == "reject"


def test_model_receives_one_row_with_the_model_column_order(make_app, database, monkeypatch):
    model = FakeModel()
    Registry(monkeypatch, champion(model, features=list(reversed(FEATURES))))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).status_code == 200
    (frame,) = model.frames
    assert list(frame.columns) == list(reversed(FEATURES))
    assert len(frame) == 1
    assert frame["loan_amnt"].dtype == np.float64
    assert frame["open_acc"].dtype == np.float64
    assert frame["home_ownership"].dtype == object
    assert frame["purpose"].dtype == object
    assert math.isnan(frame["purpose"].iloc[0])


def test_missing_optional_values_are_nan_for_the_model(make_app, database, monkeypatch):
    model = FakeModel()
    Registry(monkeypatch, champion(model))
    with TestClient(make_app()) as client:
        body = {key: value for key, value in VALID.items() if key != "open_acc"}
        assert client.post("/predict", json=body).status_code == 200
    frame = model.frames[0]
    assert math.isnan(frame["open_acc"].iloc[0])
    assert database.rows[0]["input_data"]["open_acc"] is None


def test_a_real_scikit_learn_pipeline_is_served(make_app, database, monkeypatch):
    # Pipeline proche de ce que livrera P2 : imputation, encodage, régression logistique.
    pd = pytest.importorskip("pandas")
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    numeric = ["loan_amnt", "annual_inc", "open_acc"]
    categorical = ["home_ownership", "purpose"]
    train = pd.DataFrame(
        {
            "loan_amnt": [5000.0, 30000.0, 12000.0, 35000.0],
            "annual_inc": [90000.0, 30000.0, 60000.0, 25000.0],
            "open_acc": [8.0, np.nan, 5.0, 2.0],
            "home_ownership": ["MORTGAGE", "RENT", "OWN", "RENT"],
            "purpose": pd.Series(["car", np.nan, "house", "debt"], dtype=object),
        }
    )
    pipeline = make_pipeline(
        ColumnTransformer(
            [
                ("num", SimpleImputer(), numeric),
                (
                    "cat",
                    make_pipeline(
                        SimpleImputer(strategy="constant", fill_value="absent"), OneHotEncoder(handle_unknown="ignore")
                    ),
                    categorical,
                ),
            ]
        ),
        LogisticRegression(),
    ).fit(train, [0, 1, 0, 1])
    Registry(monkeypatch, champion(pipeline))
    with TestClient(make_app()) as client:
        response = client.post("/predict", json={"loan_amnt": 30000, "annual_inc": 30000, "home_ownership": "RENT"})
    assert response.status_code == 200, response.text
    probability = response.json()["default_probability"]
    assert 0 <= probability <= 1
    assert database.rows[0]["prediction"] == probability


def test_positive_class_is_read_from_classes(make_app, database, monkeypatch):
    model = FakeModel(lambda frame: np.array([[0.7, 0.3]]))
    model.classes_ = np.array([1, 0])  # la classe 1 est en première colonne
    Registry(monkeypatch, champion(model))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).json()["default_probability"] == pytest.approx(0.7)


def test_model_without_classes_is_refused(make_app, database, monkeypatch, caplog):
    # Sans classes_, rien ne dit quelle colonne est le défaut : pas de colonne supposée.
    model = FakeModel(0.4)
    model.classes_ = None
    Registry(monkeypatch, champion(model))
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 503
    assert "classes_" in caplog.text


def test_asynchronous_load_champion_is_refused_without_warning(make_app, database, monkeypatch, caplog, recwarn):
    async def load():
        return champion()

    Registry(monkeypatch, load)
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 503
    assert "load_champion() doit être une fonction synchrone" in caplog.text
    assert not [w for w in recwarn if "never awaited" in str(w.message)]


@pytest.mark.parametrize(
    "features", [tuple(FEATURES), np.array(FEATURES), __import__("pandas").Index(FEATURES)]
)
def test_ordered_feature_sequences_are_accepted(make_app, database, monkeypatch, features):
    Registry(monkeypatch, champion(features=features))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).status_code == 200


def test_prediction_logs_contain_no_submitted_value(client, caplog):
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        assert client.post("/predict", json={**VALID, "purpose": SECRET}).status_code == 200
    assert SECRET not in caplog.text
    assert "12000" not in caplog.text
    assert "modèle credit-default version 3" in caplog.text


def test_health_reports_the_served_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == {"name": "credit-default", "version": "3", "decision_threshold": 0.31}


def test_champion_is_loaded_once_and_kept_in_memory(make_app, database, registry):
    with TestClient(make_app()) as client:
        for _ in range(5):
            assert client.post("/predict", json=VALID).status_code == 200
    assert registry.calls == 1


@pytest.mark.parametrize(("raw", "expected"), [(3, "3"), ("12", "12"), (" 4 ", "4"), (np.int64(5), "5")])
def test_version_is_written_as_text(make_app, database, monkeypatch, raw, expected):
    Registry(monkeypatch, champion(version=raw))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).json()["model_version"] == expected


def test_empty_attribute_falls_back_to_the_alternative_name(make_app, database, monkeypatch):
    Registry(monkeypatch, champion(version=None, model_version="5"))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).json()["model_version"] == "5"


@pytest.mark.parametrize("classes", [[0, 1, 2], [0, 2], [1], ["0", "1"], [0.0, 2.0]])
def test_model_without_exactly_two_classes_0_and_1_is_refused(make_app, database, monkeypatch, caplog, classes):
    model = FakeModel()
    model.classes_ = np.array(classes)
    Registry(monkeypatch, champion(model))
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 503
    assert "classes_" in caplog.text


def test_nul_character_is_refused_by_validation_and_not_recorded(client, database):
    response = client.post("/predict", json={**VALID, "purpose": "voiture" + chr(0)})
    assert response.status_code == 422
    assert response.json()["errors"] == [{"field": "purpose", "message": "doit être une chaîne de caractères valide"}]
    assert database.rows == []


@pytest.mark.parametrize("value", [2**53 + 1, -(2**53) - 1, 10**400])
def test_integer_too_large_for_the_model_is_refused_by_validation(client, database, value):
    response = client.post("/predict", json={**VALID, "open_acc": value})
    assert response.status_code == 422
    assert response.json()["errors"][0]["field"] == "open_acc"
    assert database.rows == []


def test_champion_given_as_an_object_is_accepted(make_app, database, monkeypatch):
    found = types.SimpleNamespace(
        model=FakeModel(), model_name="autre-modele", model_version=7, decision_threshold=0.5, feature_names=FEATURES
    )
    Registry(monkeypatch, found)
    with TestClient(make_app()) as client:
        body = client.post("/predict", json=VALID).json()
    assert (body["model_name"], body["model_version"], body["decision_threshold"]) == ("autre-modele", "7", 0.5)


def test_model_name_defaults_to_the_environment_then_credit_default(make_app, database, monkeypatch):
    Registry(monkeypatch, champion(name=None))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).json()["model_name"] == "credit-default"
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "modele-env")
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).json()["model_name"] == "modele-env"
    Registry(monkeypatch, champion(name="autre-modele", version="7"))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).json()["model_name"] == "autre-modele"
    # La ligne enregistrée porte le nom du modèle servi, pas une valeur fixe (A4).
    assert [(row["model_name"], row["model_version"]) for row in database.rows] == [
        ("credit-default", "3"),
        ("modele-env", "3"),
        ("autre-modele", "7"),
    ]


# --- sans modèle : 503 -------------------------------------------------------------


def test_without_champion_the_answer_is_503_with_the_documented_message(make_app, database, monkeypatch):
    Registry(monkeypatch, None)
    with TestClient(make_app()) as client:
        response = client.post("/predict", json=VALID)
        health = client.get("/health").json()
    assert response.status_code == 503
    assert response.json() == doc_error_example()
    assert response.json()["detail"] == predict.MODEL_UNAVAILABLE_DETAIL
    assert health["model"] is None and health["status"] == "degraded"


def test_without_champion_the_failed_inference_is_recorded(make_app, database, monkeypatch):
    Registry(monkeypatch, None)
    with TestClient(make_app()) as client:
        client.post("/predict", json=VALID)
    (row,) = database.rows
    assert row["status"] == "error"
    assert row["error_message"] == predict.NO_MODEL_CAUSE
    assert row["prediction"] is None and row["decision"] is None and row["model_version"] is None
    assert set(row["input_data"]) == set(FEATURES)
    assert row["latency_ms"] >= 0


def test_missing_registry_module_means_no_champion_with_a_clear_warning(make_app, database, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, served_model.CHAMPION_MODULE, None)  # import impossible
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            response = client.post("/predict", json=VALID)
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"
    assert "src.registry.champion introuvable (tâche R1 non livrée)" in caplog.text


def test_missing_dependency_inside_the_registry_is_a_load_error(make_app, database, monkeypatch, caplog):
    def broken():
        raise ModuleNotFoundError("No module named 'mlflow_absent'", name="mlflow_absent")

    Registry(monkeypatch, broken)
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 503
    assert "Chargement du @champion impossible (ModuleNotFoundError)" in caplog.text


def test_route_without_lifespan_answers_503(make_app, database, registry):
    client = TestClient(make_app())  # sans with : le lifespan ne s'exécute pas
    response = client.post("/predict", json=VALID)
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"


def test_failing_load_champion_keeps_the_api_up_without_its_message(make_app, database, monkeypatch, caplog):
    Registry(monkeypatch, RuntimeError(f"MLflow injoignable {SECRET}"))
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.get("/health").status_code == 200
            assert client.post("/predict", json=VALID).status_code == 503
    assert "Chargement du @champion impossible (RuntimeError)" in caplog.text
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"features": FEATURES[:-1]}, "absentes du modèle : purpose"),
        ({"features": [*FEATURES, "grade"]}, "absentes du contrat : grade"),
        ({"features": [*FEATURES, "loan_amnt"]}, "deux fois"),
        ({"features": []}, "liste non vide"),
        ({"features": "loan_amnt"}, "liste de noms"),
        ({"features": dict.fromkeys(FEATURES, 1)}, "liste de noms"),
        ({"features": set(FEATURES)}, "liste de noms"),
        ({"version": np.bool_(True)}, "version"),
        ({"version": 2.0}, "version"),
        ({"version": "champion"}, "jamais l'alias"),
        ({"version": 0}, "entier positif"),
        ({"version": True}, "version"),
        ({"version": None}, "version"),
        ({"threshold": 1.5}, "decision_threshold"),
        ({"threshold": -0.1}, "decision_threshold"),
        ({"threshold": float("nan")}, "decision_threshold"),
        ({"threshold": True}, "decision_threshold"),
        ({"threshold": "0.3x"}, "decision_threshold"),
        ({"threshold": None}, "decision_threshold"),
        ({"name": ""}, None),
        ({"name": 3}, "nom du modèle"),
    ],
)
def test_champion_that_cannot_be_served_gives_503(make_app, database, monkeypatch, caplog, change, reason):
    Registry(monkeypatch, champion(**change))
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            response = client.post("/predict", json=VALID)
            health = client.get("/health").json()
    if change == {"name": ""}:
        # Nom vide : repli sur le nom par défaut, le champion reste servi.
        assert response.status_code == 200
        return
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"
    assert health["model"] is None
    assert "Le @champion ne peut pas être servi" in caplog.text
    assert reason in caplog.text


def test_model_without_predict_proba_is_refused(make_app, database, monkeypatch, caplog):
    Registry(monkeypatch, champion(types.SimpleNamespace(predict=lambda frame: [0])))
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 503
    assert "predict_proba" in caplog.text


def test_model_whose_classes_lack_the_default_class_is_refused(make_app, database, monkeypatch, caplog):
    model = FakeModel()
    model.classes_ = np.array(["0", "1"])
    Registry(monkeypatch, champion(model))
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 503
    assert "exactement 0 et 1" in caplog.text


def test_champion_without_model_is_refused(make_app, database, monkeypatch):
    Registry(monkeypatch, champion(model=None) | {"model": None})
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).status_code == 503


# --- erreurs du modèle : 500 et ligne en erreur -------------------------------------


def test_model_error_gives_500_and_an_error_row_with_the_version(make_app, database, monkeypatch, caplog):
    Registry(monkeypatch, champion(FakeModel(error=ValueError(f"colonne {SECRET}"))))
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        with TestClient(make_app()) as client:
            response = client.post("/predict", json=VALID)
    assert response.status_code == 500
    assert response.json() == {"detail": predict.MODEL_FAILURE_DETAIL, "code": "INTERNAL_ERROR", "errors": None}
    (row,) = database.rows
    assert row["status"] == "error"
    assert row["model_version"] == "3" and row["model_name"] == "credit-default"
    assert row["decision_threshold"] == 0.31
    assert row["prediction"] is None and row["decision"] is None
    assert row["error_message"] == "Le modèle a échoué pendant la prédiction (ValueError)."
    assert SECRET not in caplog.text
    assert SECRET not in response.text


@pytest.mark.parametrize(
    "output",
    [
        np.array([[0.5, 1.5]]),
        np.array([[0.5, -0.1]]),
        np.array([[0.5, np.nan]]),
        np.array([[0.5, np.inf]]),
        np.array([0.2, 0.8]),
        np.array([[0.2, 0.8], [0.3, 0.7]]),
        np.array([[0.2]]),
        np.array([[0.1, 0.2, 0.7]]),
        np.array([["a", "b"]]),
    ],
)
def test_unusable_model_output_gives_500_and_an_error_row(make_app, database, monkeypatch, output):
    Registry(monkeypatch, champion(FakeModel(lambda frame: output)))
    with TestClient(make_app()) as client:
        response = client.post("/predict", json=VALID)
    assert response.status_code == 500
    (row,) = database.rows
    assert row["status"] == "error"
    assert row["error_message"].startswith("Sortie du modèle inexploitable")


def test_model_error_is_answered_even_if_the_database_is_down(make_app, database, monkeypatch, caplog):
    Registry(monkeypatch, champion(FakeModel(error=RuntimeError("panne"))))
    with TestClient(make_app()) as client:
        database.up = False
        with caplog.at_level(logging.WARNING, logger="neuraxion"):
            response = client.post("/predict", json=VALID)
    assert response.status_code == 500
    assert "en erreur non enregistrée" in caplog.text


# --- base de données -----------------------------------------------------------------


def test_schema_is_created_at_startup(make_app, database, registry):
    with TestClient(make_app()):
        pass
    assert database.schema_runs == 1
    assert database.commits >= 1


def test_database_down_at_startup_does_not_stop_the_api(make_app, database, registry, caplog):
    database.up = False
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.get("/health").status_code == 200
            database.up = True
            assert client.post("/predict", json=VALID).status_code == 200
            assert client.post("/predict", json=VALID).status_code == 200
    assert "Table inferences non créée" in caplog.text
    # Table créée à la première inférence, une seule fois.
    assert database.schema_runs == 1
    assert len(database.rows) == 2


def test_unconfigured_database_at_startup_is_reported(make_app, database, registry, caplog):
    database.up = DatabaseNotConfigured("Variables PostgreSQL non renseignées : POSTGRES_USER.")
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()):
            pass
    assert "base non configurée (Variables PostgreSQL non renseignées" in caplog.text


def test_invalid_port_does_not_stop_the_api(make_app, registry, monkeypatch, caplog):
    # Sans fausse base : la vraie connexion lit POSTGRES_PORT.
    for key, value in {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d", "POSTGRES_PORT": "54x2"}.items():
        monkeypatch.setenv(key, value)
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.get("/health").status_code == 200
            response = client.post("/predict", json=VALID)
    assert response.status_code == 503
    assert response.json()["code"] == "DATABASE_UNAVAILABLE"
    assert "POSTGRES_PORT doit être un entier" in caplog.text


def test_insert_is_tried_before_create_for_a_role_without_create_right(make_app, database, registry):
    # Table créée par le propriétaire ; le rôle de l'API peut insérer mais pas créer.
    database.table_exists = True
    database.create_error = psycopg.errors.InsufficientPrivilege("permission denied for schema public")
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).status_code == 200
        assert client.post("/predict", json=VALID).status_code == 200
    assert len(database.rows) == 2


@pytest.mark.parametrize(
    "error",
    [psycopg.OperationalError("connexion refusée"), psycopg.InterfaceError("fermée"), DatabaseNotConfigured("x")],
)
def test_database_down_during_a_prediction_gives_503(make_app, database, registry, caplog, error):
    with TestClient(make_app()) as client:
        database.up = error
        with caplog.at_level(logging.DEBUG, logger="neuraxion"):
            response = client.post("/predict", json={**VALID, "purpose": SECRET})
    assert response.status_code == 503
    assert response.json() == {
        "detail": predict.DATABASE_UNAVAILABLE_DETAIL,
        "code": "DATABASE_UNAVAILABLE",
        "errors": None,
    }
    assert SECRET not in caplog.text
    assert "non enregistrée" in caplog.text


def test_insert_failure_after_connection_gives_503(make_app, database, registry):
    with TestClient(make_app()) as client:
        database.insert_error = psycopg.OperationalError("coupure")
        assert client.post("/predict", json=VALID).status_code == 503


def test_rejected_row_is_an_internal_error_not_a_database_outage(make_app, database, registry):
    with TestClient(make_app()) as client:
        database.insert_error = psycopg.errors.CheckViolation("contrainte")
        response = client.post("/predict", json=VALID)
    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"


def test_dropped_table_is_recreated_once(make_app, database, registry):
    with TestClient(make_app()) as client:
        database.table_exists = False  # table supprimée depuis le démarrage
        assert client.post("/predict", json=VALID).status_code == 200
    assert database.rollbacks == 1
    assert database.schema_runs == 2
    assert len(database.rows) == 1


def test_without_champion_and_without_database_the_answer_stays_model_unavailable(make_app, database, monkeypatch):
    Registry(monkeypatch, None)
    database.up = False
    with TestClient(make_app()) as client:
        response = client.post("/predict", json=VALID)
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"


# --- rechargement (raccord avec A3) --------------------------------------------------


def test_reload_model_serves_the_new_champion(make_app, database, monkeypatch):
    registry = Registry(monkeypatch, champion(version="3", threshold=0.31))
    with TestClient(make_app()) as client:
        app = client.app
        assert client.post("/predict", json=VALID).json()["model_version"] == "3"
        registry.current = champion(FakeModel(0.5), version="4", threshold=0.6)
        info = app.state.reload_model()
        assert (info.version, info.decision_threshold) == ("4", 0.6)
        body = client.post("/predict", json=VALID).json()
        health = client.get("/health").json()
    assert (body["model_version"], body["decision"]) == ("4", "approve")
    assert health["model"]["version"] == "4"
    assert [row["model_version"] for row in database.rows] == ["3", "4"]


def test_reload_model_keeps_the_served_model_when_loading_fails(make_app, database, monkeypatch, caplog):
    registry = Registry(monkeypatch, champion(version="3"))
    with TestClient(make_app()) as client:
        app = client.app
        registry.current = RuntimeError("MLflow arrêté")
        with caplog.at_level(logging.WARNING, logger="neuraxion"):
            assert app.state.reload_model().version == "3"
        assert client.post("/predict", json=VALID).json()["model_version"] == "3"
        assert client.get("/health").json()["model"]["version"] == "3"
    assert "Le modèle déjà servi est gardé (credit-default version 3)" in caplog.text


def test_reload_model_with_an_incompatible_champion_stops_serving(make_app, database, monkeypatch, caplog):
    # Garde-fou du plan : le registre désigne la version 5, que l'API ne sait pas servir.
    registry = Registry(monkeypatch, champion(version="3"))
    with TestClient(make_app()) as client:
        app = client.app
        registry.current = champion(features=FEATURES[:2], version="5")
        with caplog.at_level(logging.WARNING, logger="neuraxion"):
            assert app.state.reload_model() is None
        assert client.post("/predict", json=VALID).status_code == 503
        assert client.get("/health").json()["model"] is None
    assert "absentes du modèle" in caplog.text


def test_replayed_lifespan_does_not_keep_a_stale_model_info(make_app, database, monkeypatch):
    registry = Registry(monkeypatch, champion(version="3"))
    app = make_app()
    with TestClient(app) as client:
        assert client.get("/health").json()["model"]["version"] == "3"
    registry.current = RuntimeError("MLflow arrêté")
    with TestClient(app) as client:
        assert client.get("/health").json()["model"] is None
        assert client.post("/predict", json=VALID).status_code == 503


def test_reload_model_without_champion_stops_serving(make_app, database, monkeypatch):
    registry = Registry(monkeypatch, champion())
    with TestClient(make_app()) as client:
        app = client.app
        registry.current = None
        assert app.state.reload_model() is None
        assert client.post("/predict", json=VALID).status_code == 503
        assert client.get("/health").json()["model"] is None


def test_reload_model_after_a_first_promotion(make_app, database, monkeypatch):
    registry = Registry(monkeypatch, None)
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).status_code == 503
        registry.current = champion(version="1")
        assert client.app.state.reload_model().version == "1"
        assert client.post("/predict", json=VALID).status_code == 200


def test_reload_model_is_safe_under_concurrent_calls(make_app, database, monkeypatch):
    import threading
    import time

    state = {"running": 0, "most": 0, "count": 0}
    lock = threading.Lock()

    def slow_load():
        with lock:
            state["running"] += 1
            state["most"] = max(state["most"], state["running"])
            state["count"] += 1
            version = str(state["count"])
        time.sleep(0.02)  # laisse aux autres appels le temps d'entrer s'ils le peuvent
        with lock:
            state["running"] -= 1
        return champion(version=version)

    Registry(monkeypatch, slow_load)
    with TestClient(make_app()) as client:
        app = client.app
        threads = [threading.Thread(target=app.state.reload_model) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        served, info = app.state.served_model, app.state.model_info
        body = client.post("/predict", json=VALID).json()
    assert state["most"] == 1  # jamais deux chargements en même temps
    assert state["count"] == 9
    assert served.version == info.version == body["model_version"] == "9"


# --- contrat et documentation --------------------------------------------------------


def test_openapi_lists_the_predict_errors(client):
    responses = client.get("/openapi.json").json()["paths"]["/predict"]["post"]["responses"]
    assert {"200", "422", "500", "503"} <= set(responses)
    assert "501" not in responses


def test_categorical_fields_follow_the_contract():
    model = build_request_model(CONTRACT)
    assert predict.categorical_fields(model) == {"home_ownership", "purpose"}


def test_error_details_are_valid_error_responses():
    for detail in (predict.MODEL_UNAVAILABLE_DETAIL, predict.DATABASE_UNAVAILABLE_DETAIL, predict.MODEL_FAILURE_DETAIL):
        ErrorResponse(detail=detail, code="INTERNAL_ERROR")


def test_doc_describes_reload_model_and_the_r1_connection():
    assert "app.state.reload_model" in DOC_TEXT
    assert "src.registry.champion" in DOC_TEXT
    assert "| `POST /predict` | prédire le risque de défaut d'une demande | A2 | 200 |" in DOC_TEXT


# --- src/db : enregistrement et création de la table ------------------------------


def _record(**change):
    values = {
        "id": UUID("6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d"),
        "created_at": datetime.fromisoformat("2026-11-27T10:15:42.120000+01:00"),
        "input_data": {"loan_amnt": 12000.0},
        "status": "success",
        "latency_ms": 3,
        "prediction": 0.23,
        "decision": "approve",
        "decision_threshold": 0.31,
        "model_name": "credit-default",
        "model_version": "3",
    }
    values.update(change)
    return inferences.InferenceRecord(**values)


def test_save_inference_sends_every_column():
    database = FakeDatabase()
    database.table_exists = True
    assert inferences.save_inference(_record(), FakeConnection(database)) == _record().id
    (row,) = database.rows
    assert set(row) == {
        "id",
        "created_at",
        "input_data",
        "prediction",
        "decision",
        "decision_threshold",
        "model_name",
        "model_version",
        "status",
        "error_message",
        "latency_ms",
    }
    assert row["input_data"] == {"loan_amnt": 12000.0}


def test_save_inference_opens_and_commits_its_own_connection(monkeypatch):
    database = FakeDatabase()
    database.table_exists = True
    monkeypatch.setattr(inferences, "connect", database.connect)
    inferences.save_inference(_record())
    assert len(database.rows) == 1 and database.commits == 1


@pytest.mark.parametrize(
    "change",
    [
        {"created_at": datetime(2026, 11, 27, 10, 15)},
        {"status": "ok"},
        {"latency_ms": -1},
        {"latency_ms": 1.5},
        {"latency_ms": True},
        {"prediction": 1.2},
        {"prediction": float("nan")},
        {"decision_threshold": -0.5},
    ],
)
def test_inference_record_refuses_impossible_values(change):
    with pytest.raises(ValueError):
        _record(**change)


def test_create_schema_runs_the_sql_file_under_an_advisory_lock():
    database = FakeDatabase()
    init_db.create_schema(FakeConnection(database))
    lock, script = database.queries
    assert "pg_advisory_xact_lock" in lock
    assert script == init_db.SCHEMA_PATH.read_text(encoding="utf-8")


def test_init_db_command_reports_success(monkeypatch, capsys):
    database = FakeDatabase()
    monkeypatch.setattr(init_db, "connect", database.connect)
    assert init_db.main() == 0
    assert "Table inferences prête" in capsys.readouterr().out
    assert database.schema_runs == 1


def test_init_db_command_without_configuration_exits_1(capsys):
    assert init_db.main() == 1  # conftest retire les variables POSTGRES_*
    assert "POSTGRES_USER" in capsys.readouterr().err


def test_init_db_command_with_database_down_prints_only_the_error_type(monkeypatch, capsys):
    def refused(environ=None):
        raise psycopg.OperationalError(f"mot de passe {SECRET} refusé pour l'hôte db")

    monkeypatch.setattr(init_db, "connect", refused)
    assert init_db.main() == 1
    err = capsys.readouterr().err
    assert "OperationalError" in err
    assert SECRET not in err


def test_init_db_runs_as_a_module():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "src.db.init_db"],
        cwd=REPO_ROOT,
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        env={key: value for key, value in __import__("os").environ.items() if not key.startswith("POSTGRES_")}
        | {"PYTHONIOENCODING": "utf-8"},
        check=False,
    )
    assert result.returncode == 1
    assert "Schéma non créé" in result.stderr


def test_init_db_command_with_an_invalid_port_exits_1_with_a_clear_message(monkeypatch, capsys):
    for key, value in {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d", "POSTGRES_PORT": "54x2"}.items():
        monkeypatch.setenv(key, value)
    assert init_db.main() == 1
    assert "POSTGRES_PORT doit être un entier" in capsys.readouterr().err


def test_connection_params_refuse_a_non_integer_port():
    from src.db import connection

    environ = {"POSTGRES_USER": "u", "POSTGRES_PASSWORD": "p", "POSTGRES_DB": "d", "POSTGRES_PORT": "54x2"}
    with pytest.raises(DatabaseNotConfigured, match="POSTGRES_PORT"):
        connection.connection_params(environ)
    assert connection.database_is_up(environ) is False


# --- constats de la boucle 2 (revue Codex) -------------------------------------------


def test_contract_refuses_a_category_with_a_nul_character():
    contract = {"features": [{"name": "statut", "type": "categorical", "allowed": ["ok", "nul" + chr(0)]}]}
    with pytest.raises(ValueError, match="NUL"):
        build_request_model(contract)


def test_contract_refuses_an_integer_category_beyond_2_53():
    contract = {"features": [{"name": "code", "type": "integer", "allowed": [1, 2**53 + 1]}]}
    with pytest.raises(ValueError, match="trop grande"):
        build_request_model(contract)


@pytest.mark.parametrize("required", [True, False])
def test_integer_bounds_stay_in_the_published_schema(required):
    contract = {"features": [{"name": "n", "type": "integer", "min": 1, "max": 8, "required": required}]}
    prop = build_request_model(contract).model_json_schema()["properties"]["n"]
    branch = prop if required else next(b for b in prop["anyOf"] if b.get("type") == "integer")
    assert (branch["minimum"], branch["maximum"]) == (1, 8)
    assert not {"ge", "le"} & set(branch)


@pytest.mark.parametrize("position", ["start", "middle", "end"])
def test_nul_character_is_refused_anywhere_in_the_text(client, database, position):
    text = {"start": chr(0) + "ab", "middle": "a" + chr(0) + "b", "end": "ab" + chr(0)}[position]
    assert client.post("/predict", json={**VALID, "purpose": text}).status_code == 422
    assert database.rows == []


@pytest.mark.parametrize("value", [2**53, -(2**53)])
def test_integers_at_2_53_are_accepted(client, database, value):
    assert client.post("/predict", json={**VALID, "open_acc": abs(value)}).status_code == 200
    model = build_request_model(
        {"features": [{"name": "n", "type": "integer"}]}
    )
    assert model.model_validate({"n": value}).n == value


@pytest.mark.parametrize(
    "change",
    [
        {"model_classes": [None, 1]},
        {"model_classes": [[0], [1]]},
        {"features": np.array("loan_amnt")},
        {"threshold": 10**400},
        {"threshold": np.bool_(True)},
    ],
)
def test_reload_with_an_unusable_result_stops_serving(make_app, database, monkeypatch, caplog, change):
    registry = Registry(monkeypatch, champion(version="3"))
    with TestClient(make_app()) as client:
        app = client.app
        model = FakeModel()
        if "model_classes" in change:
            model.classes_ = np.array(change.pop("model_classes"), dtype=object)
        registry.current = champion(model, version="4", **change)
        with caplog.at_level(logging.WARNING, logger="neuraxion"):
            assert app.state.reload_model() is None
        assert client.post("/predict", json=VALID).status_code == 503
        assert client.get("/health").json()["model"] is None
    assert "Le @champion ne peut pas être servi" in caplog.text


def test_exception_raised_by_load_champion_keeps_the_model_whatever_its_type(make_app, database, monkeypatch, caplog):
    registry = Registry(monkeypatch, champion(version="3"))
    with TestClient(make_app()) as client:
        app = client.app
        registry.current = served_model.ChampionError(f"panne R1 {SECRET}")
        with caplog.at_level(logging.DEBUG, logger="neuraxion"):
            assert app.state.reload_model().version == "3"
        assert client.post("/predict", json=VALID).json()["model_version"] == "3"
    assert "Chargement du @champion impossible (ChampionError)" in caplog.text
    assert SECRET not in caplog.text


def test_message_of_an_exception_raised_by_the_model_never_reaches_the_database(make_app, database, monkeypatch, caplog):
    Registry(monkeypatch, champion(FakeModel(error=served_model.ModelOutputError(f"témoin {SECRET}"))))
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 500
    (row,) = database.rows
    assert row["error_message"] == "Le modèle a échoué pendant la prédiction (ModelOutputError)."
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    "output",
    [np.array([[0.4, 0.6 + 0.7j]]), np.array([[False, True]]), np.array([[0.2, 0.3, 0.4, 0.1]]), [[0.5, None]]],
)
def test_non_real_or_misshaped_outputs_are_refused(make_app, database, monkeypatch, output):
    Registry(monkeypatch, champion(FakeModel(lambda frame: output)))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).status_code == 500
    assert database.rows[0]["error_message"].startswith("Sortie du modèle inexploitable")


@pytest.mark.parametrize("classes", [[True, False], [1.0, 0.0]])
def test_boolean_or_float_labels_find_the_default_column(make_app, database, monkeypatch, classes):
    model = FakeModel(lambda frame: np.array([[0.6, 0.4]]))
    model.classes_ = np.array(classes)
    Registry(monkeypatch, champion(model))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).json()["default_probability"] == pytest.approx(0.6)


def test_table_dropped_again_after_its_creation_is_an_error_without_row(make_app, database, registry):
    with TestClient(make_app()) as client:
        database.table_exists = False
        original = FakeConnection.execute

        def drop_after_create(self, query, params=None):
            original(self, query, params)
            if "CREATE TABLE" in query:
                self.database.table_exists = False  # suppression concurrente juste après la création

        FakeConnection.execute = drop_after_create
        try:
            response = client.post("/predict", json=VALID)
        finally:
            FakeConnection.execute = original
    assert response.status_code == 500
    assert database.rows == []
    assert database.queries.count(inferences.INSERT_INFERENCE) == 2


# --- tests proposés par l'injection d'erreurs (boucle 1) -----------------------------


class Clock:
    """Remplace le module time de predict : perf_counter renvoie les valeurs données, la dernière ensuite."""

    def __init__(self, *values):
        self.values = list(values)

    def perf_counter(self):
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


# Tue M17, M18, M93 : latence nulle, en secondes, ou non transmise à la base.
def test_latency_is_measured_in_milliseconds(make_app, database, registry, monkeypatch):
    with TestClient(make_app()) as client:
        monkeypatch.setattr(predict, "time", Clock(100.0, 100.25))
        assert client.post("/predict", json=VALID).status_code == 200
    assert database.rows[0]["latency_ms"] == 250


# Tue M23 : le log « base injoignable » ne recopie pas le message de psycopg.
def test_database_outage_log_omits_the_exception_message(make_app, database, registry, caplog):
    with TestClient(make_app()) as client:
        database.up = psycopg.OperationalError(f"connexion refusée pour {SECRET}")
        with caplog.at_level(logging.DEBUG, logger="neuraxion"):
            assert client.post("/predict", json=VALID).status_code == 503
    assert SECRET not in caplog.text
    assert "(OperationalError)" in caplog.text


# Tue M22 : l'échec d'enregistrement d'une ligne en erreur ne recopie pas le message.
def test_failed_error_row_log_omits_the_exception_message(make_app, database, monkeypatch, caplog):
    Registry(monkeypatch, None)
    with TestClient(make_app()) as client:
        database.up = psycopg.OperationalError(f"hôte {SECRET}")
        with caplog.at_level(logging.DEBUG, logger="neuraxion"):
            assert client.post("/predict", json=VALID).status_code == 503
    assert SECRET not in caplog.text
    assert "en erreur non enregistrée" in caplog.text


# Tue M35 : le log de création de la table au démarrage ne recopie pas le message.
def test_startup_schema_log_omits_the_exception_message(make_app, database, registry, caplog):
    database.up = psycopg.OperationalError(f"mot de passe {SECRET} refusé")
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        with TestClient(make_app()):
            pass
    assert SECRET not in caplog.text
    assert "Table inferences non créée" in caplog.text


# Tue M34 : un script refusé (ProgrammingError) au démarrage n'arrête pas l'API.
def test_schema_refused_at_startup_does_not_stop_the_api(make_app, database, registry, caplog):
    database.up = psycopg.errors.InsufficientPrivilege("permission refusée")
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.get("/health").status_code == 200
    assert "Table inferences non créée" in caplog.text


# Tue M80 : init_db rattrape toute erreur psycopg (script refusé), code 1, sans le message.
def test_init_db_command_with_refused_script_exits_1(monkeypatch, capsys):
    def refused(environ=None):
        raise psycopg.errors.InsufficientPrivilege(f"permission refusée pour {SECRET}")

    monkeypatch.setattr(init_db, "connect", refused)
    assert init_db.main() == 1
    err = capsys.readouterr().err
    assert "InsufficientPrivilege" in err
    assert SECRET not in err


# Tue M42 : sans modèle, une ligne en erreur refusée par la base garde la réponse 503.
def test_without_champion_a_rejected_error_row_keeps_model_unavailable(make_app, database, monkeypatch):
    Registry(monkeypatch, None)
    with TestClient(make_app()) as client:
        database.insert_error = psycopg.errors.CheckViolation("contrainte")
        response = client.post("/predict", json=VALID)
    assert response.status_code == 503
    assert response.json()["code"] == "MODEL_UNAVAILABLE"


# Tue M50 : paquet src.registry absent = pas de champion, avec l'avertissement prévu.
def test_missing_registry_package_means_no_champion(make_app, database, monkeypatch, caplog):
    # Ce que lève Python quand le dossier src/registry n'existe pas (name = "src.registry").
    def import_module(name):
        raise ModuleNotFoundError("No module named 'src.registry'", name="src.registry")

    monkeypatch.setattr(served_model, "importlib", types.SimpleNamespace(import_module=import_module))
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 503
    assert "src.registry.champion introuvable (tâche R1 non livrée)" in caplog.text


# Tue M88 : chaque colonne du INSERT reçoit le paramètre de même nom.
def test_insert_columns_and_placeholders_match():
    columns_part, values_part = inferences.INSERT_INFERENCE.split("VALUES")
    columns = re.findall(r"\w+", columns_part.split("(", 1)[1])
    placeholders = re.findall(r"%\((\w+)\)s", values_part)
    assert columns == placeholders
    assert set(placeholders) == set(_record().as_parameters())


class RecordingConnection(FakeConnection):
    def transaction(self):
        self.database.queries.append("BEGIN")
        return self

    def __exit__(self, kind, error, trace):
        self.database.queries.append("COMMIT" if error is None else "ROLLBACK")
        return False


# Tue M74 : le verrou et le script sont exécutés dans une transaction explicite.
def test_create_schema_runs_inside_its_own_transaction():
    database = FakeDatabase()
    init_db.create_schema(RecordingConnection(database))
    assert database.queries[0] == "BEGIN"
    assert "pg_advisory_xact_lock" in database.queries[1]
    assert database.queries[2] == init_db.SCHEMA_PATH.read_text(encoding="utf-8")
    assert database.queries[3] == "COMMIT"


# Tue M48 : une dépendance absente à l'import de R1 (mlflow) est une erreur de chargement.
def test_missing_dependency_at_registry_import_is_a_load_error(make_app, database, monkeypatch, caplog):
    def import_module(name):
        raise ModuleNotFoundError("No module named 'mlflow'", name="mlflow")

    monkeypatch.setattr(served_model, "importlib", types.SimpleNamespace(import_module=import_module))
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 503
    assert "Chargement du @champion impossible (ModuleNotFoundError)" in caplog.text
    assert "introuvable" not in caplog.text


# Boucle 3 : filet de adapt_champion, noms de type libres, sorties object.


class ChampionWithBrokenModel:
    """Champion dont l'attribut model lève une exception ordinaire (pas une ChampionError)."""

    version = "4"
    decision_threshold = 0.31
    features = list(FEATURES)

    @property
    def model(self):
        raise RuntimeError(f"panne {SECRET}")


def test_unexpected_error_while_adapting_the_champion_stops_serving(make_app, database, monkeypatch, caplog):
    registry = Registry(monkeypatch, champion(version="3"))
    with TestClient(make_app()) as client:
        registry.current = ChampionWithBrokenModel()
        with caplog.at_level(logging.WARNING, logger="neuraxion"):
            assert client.app.state.reload_model() is None
        assert client.post("/predict", json=VALID).status_code == 503
        assert client.get("/health").json()["model"] is None
    assert "Résultat de load_champion() inexploitable (RuntimeError)" in caplog.text
    assert SECRET not in caplog.text


def test_unexpected_error_while_adapting_the_champion_does_not_block_startup(make_app, database, monkeypatch, caplog):
    Registry(monkeypatch, ChampionWithBrokenModel())
    with caplog.at_level(logging.WARNING, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.get("/health").json()["model"] is None
            assert client.post("/predict", json=VALID).status_code == 503
    assert "inexploitable (RuntimeError)" in caplog.text


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("ValueError", "ValueError"),
        ("_Prive2", "_Prive2"),
        ("x" * 64, "x" * 64),
        ("x" * 65, "exception"),
        ("Nom libre TEMOIN", "exception"),
        ("9Debut", "exception"),
        ("Accentué", "exception"),
        ("Fin\n", "exception"),
    ],
)
def test_error_type_name_keeps_only_short_identifiers(name, expected):
    error = type(name, (Exception,), {})()
    assert served_model.error_type_name(error) == expected


def test_free_exception_class_name_never_reaches_the_database_or_logs(make_app, database, monkeypatch, caplog):
    witness = f"Nom libre {SECRET}"
    Registry(monkeypatch, champion(FakeModel(error=type(witness, (Exception,), {})())))
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        with TestClient(make_app()) as client:
            assert client.post("/predict", json=VALID).status_code == 500
    (row,) = database.rows
    assert row["error_message"] == "Le modèle a échoué pendant la prédiction (exception)."
    assert SECRET not in caplog.text


@pytest.mark.parametrize(
    "output",
    [
        np.array([[0.6, 0.4]], dtype=object),
        np.array([[Decimal("0.6"), Decimal("0.4")]], dtype=object),
        np.array([[0.6, np.float32(0.4)]], dtype=object),
    ],
)
def test_object_output_made_of_reals_is_converted(make_app, database, monkeypatch, output):
    Registry(monkeypatch, champion(FakeModel(lambda frame: output)))
    with TestClient(make_app()) as client:
        response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert response.json()["default_probability"] == pytest.approx(0.4)
    assert database.rows[0]["status"] == "success"


@pytest.mark.parametrize(
    "output",
    [
        np.array([[False, True]], dtype=object),
        np.array([[0.6, np.bool_(True)]], dtype=object),
        np.array([[0.6, None]], dtype=object),
        np.array([[0.6, "0.4"]], dtype=object),
        np.array([[0.6, 0.4j]], dtype=object),
        np.array([[0.6, Decimal("NaN")]], dtype=object),
        np.array([[0.6, Decimal("1e400")]], dtype=object),
    ],
)
def test_object_output_with_a_non_real_value_is_refused(make_app, database, monkeypatch, output):
    Registry(monkeypatch, champion(FakeModel(lambda frame: output)))
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=VALID).status_code == 500
    assert database.rows[0]["error_message"].startswith("Sortie du modèle inexploitable")

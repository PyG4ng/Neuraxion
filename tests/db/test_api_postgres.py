"""Tâche A4 : vérifications 3 et 4 du §13 sur l'application complète et un vrai PostgreSQL.

- §13 n° 3 : POST /predict avec une entrée valide, puis une entrée invalide ;
- §13 n° 4 et critère I : une prédiction réussie est retrouvable dans PostgreSQL avec la
  version du modèle, y compris après une promotion par POST /train ;
- POST /train sans clé et avec une mauvaise clé : refusé, sans entraînement, sans
  rechargement et sans ligne en base.

Les mêmes comportements sont testés sans base dans tests/api (test_predict.py, test_train.py).
Base, schéma temporaire et faux @champion : voir conftest.py. Sautés sans POSTGRES_*.
"""

import re

import pytest
from fastapi.testclient import TestClient

from src.api import main, schemas, train

KEY = "cle-a4-de-test-7Hq2"
PROMOTED = schemas.TrainResponse.model_config["json_schema_extra"]["examples"][0]
DOCUMENTED_QUERY = re.search(
    r"Pour retrouver une inférence et la version du modèle :\s*```sql\s*(.*?)```",
    (main.ROOT / "docs" / "api.md").read_text(encoding="utf-8"),
    re.S,
).group(1)


class FakeTrainer:
    """Fausse train_and_promote (R2) : promeut la version 4 dans le faux registre."""

    def __init__(self, registry):
        self.registry, self.presets = registry, []

    def __call__(self, preset):
        self.presets.append(preset)
        self.registry.serve(version="4", threshold=0.6)
        return {**PROMOTED, "model_version": "4"}


@pytest.fixture
def trainer(registry, monkeypatch):
    monkeypatch.setenv("TRAIN_API_KEY", KEY)
    return FakeTrainer(registry)


@pytest.fixture
def client(make_app, registry, trainer):
    registry.serve(version="3", threshold=0.4)
    app = make_app()
    app.dependency_overrides[train.get_train_and_promote] = lambda: trainer
    with TestClient(app) as test_client:
        yield test_client


def documented_query(database) -> list[tuple]:
    with database.connect() as conn:
        return conn.execute(DOCUMENTED_QUERY).fetchall()


def test_valid_then_invalid_request_and_the_documented_query(client, database, valid_request):
    response = client.post("/predict", json=valid_request)
    assert response.status_code == 200, response.text
    body = schemas.PredictResponse.model_validate(response.json())
    refused = client.post("/predict", json={**valid_request, "loan_amnt": 999})
    assert refused.status_code == 422
    assert refused.json()["code"] == "VALIDATION_ERROR"
    assert refused.json()["errors"][0]["field"] == "loan_amnt"
    # La requête SQL de docs/api.md retrouve la prédiction, avec la version du modèle ; l'entrée
    # refusée n'a laissé aucune ligne.
    ((found_id, created_at, prediction, decision, version),) = documented_query(database)
    assert (found_id, created_at) == (body.inference_id, body.created_at)
    assert (prediction, decision, version) == (pytest.approx(0.42), "reject", "3")
    assert body.model_version == "3"


def test_the_recorded_version_follows_a_promotion(client, database, trainer, valid_request):
    before = client.post("/predict", json=valid_request).json()
    promoted = client.post("/train", headers={"X-API-Key": KEY}, json={"preset": "rapide"})
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["model_version"] == "4"
    after = client.post("/predict", json=valid_request).json()
    assert (before["model_version"], after["model_version"]) == ("3", "4")
    assert (after["decision_threshold"], after["decision"]) == (0.6, "approve")
    assert client.get("/health").json()["model"]["version"] == "4"
    assert trainer.presets == ["rapide"]
    rows = {str(row["id"]): row for row in database.rows()}
    assert rows[before["inference_id"]]["model_version"] == "3"
    assert rows[after["inference_id"]]["model_version"] == "4"
    assert (rows[after["inference_id"]]["decision_threshold"], rows[after["inference_id"]]["decision"]) == (
        0.6,
        "approve",
    )
    assert [version for (*_, version) in documented_query(database)] == ["4", "3"]


@pytest.mark.parametrize(
    ("headers", "status", "code"),
    [({}, 401, "UNAUTHORIZED"), ({"X-API-Key": "mauvaise-cle"}, 403, "FORBIDDEN")],
    ids=["sans-cle", "mauvaise-cle"],
)
def test_train_without_key_or_with_a_wrong_key_changes_nothing(
    client, database, registry, trainer, valid_request, headers, status, code
):
    response = client.post("/train", headers=headers, json={"preset": "rapide"})
    assert response.status_code == status
    assert response.json()["code"] == code
    assert KEY not in response.text
    assert trainer.presets == []
    assert registry.loads == 1  # seul le chargement du démarrage
    assert database.rows() == []
    served = client.post("/predict", json=valid_request).json()
    assert served["model_version"] == "3"
    assert [row["model_version"] for row in database.rows()] == ["3"]


def test_recorded_inferences_survive_a_restart(make_app, registry, database, valid_request):
    registry.serve(version="3")
    with TestClient(make_app()) as client:
        first = client.post("/predict", json=valid_request).json()
    registry.serve(version="5")
    with TestClient(make_app()) as client:  # nouveau démarrage : create_schema() est relancé
        second = client.post("/predict", json=valid_request).json()
    rows = database.rows()
    assert [(str(row["id"]), row["model_version"]) for row in rows] == [
        (first["inference_id"], "3"),
        (second["inference_id"], "5"),
    ]


def test_the_tests_write_only_in_their_temporary_schema(client, database, valid_request):
    def public_rows():
        with database.connect() as conn:
            conn.execute("SET search_path TO public")
            if conn.execute("SELECT to_regclass('public.inferences')").fetchone()[0] is None:
                return None
            return conn.execute("SELECT count(*) FROM public.inferences").fetchone()[0]

    before = public_rows()
    assert client.post("/predict", json=valid_request).status_code == 200
    with database.connect() as conn:
        assert conn.execute("SELECT current_schema()").fetchone()[0] == database.schema
    assert len(database.rows()) == 1
    assert public_rows() == before

"""Tests contre un vrai PostgreSQL (dossier tests/db, tâches A2 et A4).

La base est celle des variables POSTGRES_* (src/db/connection.py), par exemple le service
postgres de docker-compose.yml vu depuis la machine : POSTGRES_HOST=127.0.0.1 et
POSTGRES_PORT=5433 (commande complète dans docs/api.md, section « Tests de l'API et de la
base (A4) »).

- Sans POSTGRES_USER, POSTGRES_PASSWORD ou POSTGRES_DB, chaque test est sauté.
- Base configurée mais injoignable : chaque test échoue, pour qu'une base arrêtée ne passe
  pas pour un succès.
- Chaque test travaille dans un schéma temporaire (neuraxion_test_...) supprimé à la fin.
  La variable PGOPTIONS place ce schéma seul dans le search_path de toutes les connexions du
  processus (API, tests, sous-processus) : la table inferences du schéma public, et donc les
  données de la base de Compose, ne sont jamais lues ni modifiées.
- Le @champion (R1) est remplacé par un faux module src.registry.champion, sans MLflow.
"""

import json
import os
import secrets
import sys
import types
from dataclasses import dataclass, field

import numpy as np
import psycopg
import pytest
import yaml
from psycopg import sql
from psycopg.rows import dict_row

from src.api import main, served_model
from src.db import connection

CONTRACT = {
    "version": "test-postgres",
    "features": [
        {"name": "loan_amnt", "type": "numeric", "min": 1000, "max": 40000},
        {"name": "annual_inc", "type": "numeric", "min": 0},
        {"name": "home_ownership", "type": "categorical", "allowed": ["RENT", "OWN", "MORTGAGE"]},
        {"name": "dti", "type": "numeric", "required": False},
        {"name": "purpose", "type": "categorical", "required": False},
    ],
}
FEATURES = [feature["name"] for feature in CONTRACT["features"]]
VALID_REQUEST = {"loan_amnt": 12000, "annual_inc": 55000, "home_ownership": "RENT"}
SCHEMA_PREFIX = "neuraxion_test_"
POSTGRES_VARIABLES = ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")


@dataclass
class Database:
    """La base de test : configuration POSTGRES_* lue au début du test et schéma temporaire."""

    environ: dict[str, str]
    schema: str

    def connect(self) -> psycopg.Connection:
        return connection.connect(self.environ)

    def rows(self, where: str = "TRUE", params: tuple = ()) -> list[dict]:
        """Lignes de inferences du schéma de test, de la plus ancienne à la plus récente."""
        with self.connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
            query = f"SELECT * FROM inferences WHERE {where} ORDER BY created_at"
            return cursor.execute(query, params).fetchall()


def _configured_environ() -> dict[str, str]:
    return {name: os.environ[name] for name in POSTGRES_VARIABLES if os.environ.get(name)}


@pytest.fixture(autouse=True)
def database(monkeypatch) -> Database:
    environ = _configured_environ()
    try:
        connection.connection_params(environ)
    except connection.DatabaseNotConfigured as error:
        pytest.skip(f"PostgreSQL non configuré : {error} Voir docs/api.md (tests A4).")
    schema = SCHEMA_PREFIX + secrets.token_hex(6)
    try:
        with connection.connect(environ) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    except psycopg.Error as error:
        # Seul le type de l'erreur : le message de psycopg peut nommer l'hôte ou l'utilisateur.
        pytest.fail(
            f"PostgreSQL configuré mais injoignable, ou schéma de test refusé ({type(error).__name__}). "
            "Depuis la machine, avec Compose : POSTGRES_HOST=127.0.0.1 et POSTGRES_PORT=5433.",
            pytrace=False,
        )
    for name, value in environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("PGOPTIONS", f"-c search_path={schema}")
    monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
    try:
        yield Database(environ, schema)
    finally:
        with connection.connect(environ) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


class FakeModel:
    """Modèle factice : probabilité fixe, ou exception levée par predict_proba."""

    classes_ = np.array([0, 1])

    def __init__(self, probability: float = 0.42, error: BaseException | None = None):
        self.probability, self.error = probability, error

    def predict_proba(self, frame):
        if self.error is not None:
            raise self.error
        return np.array([[1 - self.probability, self.probability]])


@dataclass
class Registry:
    """Faux src.registry.champion : load_champion() renvoie current (None : pas de champion)."""

    current: object = None
    loads: int = field(default=0)

    def serve(self, version="1", threshold=0.4, probability=0.42, error=None, **extra) -> dict:
        """Désigne un nouveau @champion, servi au prochain chargement ; le renvoie."""
        self.current = {
            "model": FakeModel(probability, error),
            "version": version,
            "decision_threshold": threshold,
            "features": list(FEATURES),
            **extra,
        }
        return self.current

    def load_champion(self):
        self.loads += 1
        return self.current


@pytest.fixture(autouse=True)
def registry(monkeypatch) -> Registry:
    fake = Registry()
    module = types.ModuleType(served_model.CHAMPION_MODULE)
    module.load_champion = fake.load_champion
    monkeypatch.setitem(sys.modules, served_model.CHAMPION_MODULE, module)
    return fake


@pytest.fixture
def make_app(tmp_path):
    """Fabrique l'application complète (A1, A2, A3) sur le contrat de test."""

    def factory():
        contract_path, examples_path = tmp_path / "features.yaml", tmp_path / "examples.json"
        contract_path.write_text(yaml.safe_dump(CONTRACT), encoding="utf-8")
        examples_path.write_text(json.dumps({"moyen": VALID_REQUEST}), encoding="utf-8")
        return main.create_app(contract_path, examples_path)

    return factory


@pytest.fixture
def valid_request() -> dict:
    return dict(VALID_REQUEST)

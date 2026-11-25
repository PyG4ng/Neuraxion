"""Configuration commune des tests du lot P4 (dossier tests/api).

Les tests ne contactent jamais de vraie base : les variables POSTGRES_* de la machine sont
retirées de l'environnement pendant chaque test, comme MLFLOW_MODEL_NAME. Les tests contre un
PostgreSQL réel sont dans tests/db (A2 et A4), qui lit ces variables.
Le module src.registry.champion est remplacé par un faux qui ne renvoie aucun champion.
"""

import os
import sys
import types

import pytest

POSTGRES_VARIABLES = ("POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
# Lues dans .env.production quand on lance les tests avec uv run --env-file (docs/api.md, A4).
MODEL_VARIABLES = ("MLFLOW_MODEL_NAME",)


@pytest.fixture(autouse=True)
def _no_real_database(monkeypatch):
    for name in POSTGRES_VARIABLES + MODEL_VARIABLES:
        if name in os.environ:
            monkeypatch.delenv(name)


@pytest.fixture(autouse=True)
def _no_real_champion(monkeypatch):
    # Pas de vrai @champion (MLflow) dans les tests unitaires, même une fois R1 livré : les
    # tests qui en ont besoin installent leur propre faux module.
    module = types.ModuleType("src.registry.champion")
    module.load_champion = lambda: None
    monkeypatch.setitem(sys.modules, "src.registry.champion", module)

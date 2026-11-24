"""Données de test du lot P2, faites à la main : aucun fichier de données n'est lu."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def petits_prets() -> pd.DataFrame:
    """60 prêts bruts de 2012 à 2014 (12 en 2014), avec les colonnes que reçoit run_pipeline()."""
    rng = np.random.default_rng(42)
    n = 60
    montant = rng.integers(1_000, 35_001, n).astype(float)
    mois = pd.date_range("2012-01-01", "2014-12-01", freq="MS").to_numpy()
    return pd.DataFrame(
        {
            "loan_amnt": montant,
            "annual_inc": rng.integers(20_000, 150_001, n).astype(float),
            "dti": rng.uniform(0, 35, n).round(2),
            "home_ownership": np.resize(["RENT", "OWN", "MORTGAGE"], n),
            "loan_status": np.resize(["Fully Paid", "Charged Off", "Fully Paid", "Fully Paid"], n),
            "issue_d": np.resize(mois, n),
            "funded_amnt": montant,
            "total_pymnt": (montant * rng.uniform(0.3, 1.3, n)).round(2),
        }
    )


@pytest.fixture
def petit_contrat() -> dict:
    """Contrat réduit à quatre variables. Format provisoire, à aligner sur contracts/features.yaml (C1)."""
    return {
        "version": "test",
        "features": [
            {"name": "loan_amnt", "type": "numeric"},
            {"name": "annual_inc", "type": "numeric"},
            {"name": "dti", "type": "numeric"},
            {"name": "home_ownership", "type": "categorical"},
        ],
    }

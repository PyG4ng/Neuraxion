"""Étape 1 du pipeline : cleaning. Implémentation réelle : tâche M1."""

import logging
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .config import load_config

logger = logging.getLogger(__name__)


def clean(df: pd.DataFrame, *, cfg: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Nettoie les prêts bruts et construit la cible.

    Les prêts arrivent de la table d'entraînement, de dev.parquet ou de
    make_synthetic_loans(n, seed) : variables du contrat, plus loan_status, issue_d,
    funded_amnt et total_pymnt.

    Comportement prévu (M1). La colonne target est construite à partir de loan_status
    selon la section target du protocole, et les lignes aux autres statuts sont écartées.
    Les valeurs impossibles deviennent manquantes (revenu nul ou négatif, dti négatif, etc.)
    et les doublons sont retirés. Les variables dérivées du contrat sont ajoutées, comme
    l'ancienneté du crédit en mois. issue_d devient une date et les types suivent le contrat.
    Le DataFrame reçu n'est jamais modifié.

    Corps factice (C2) : copie de df, à laquelle on ajoute seulement la cible quand
    loan_status est présent.

    Paramètres
        df : prêts bruts.
        cfg : protocole déjà chargé. Par défaut, load_config().

    Renvoie
        Un nouveau DataFrame, avec la colonne target (0 ou 1) quand loan_status est présent.
    """
    logger.warning("clean : corps factice (C2), remplacé par M1.")
    cfg = load_config() if cfg is None else cfg
    target = cfg["target"]
    out = df.copy()
    if target["source"] in out.columns and target["column"] not in out.columns:
        mapping = {status: 1 for status in target["positive"]}
        mapping.update({status: 0 for status in target["negative"]})
        out[target["column"]] = out[target["source"]].map(mapping)
        out = out.dropna(subset=[target["column"]])
        out[target["column"]] = out[target["column"]].astype("int64")
    return out

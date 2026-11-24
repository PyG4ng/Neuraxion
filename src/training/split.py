"""Découpage dans le temps défini par le protocole. Implémentation réelle : tâche M1."""

import logging
from collections.abc import Mapping
from typing import Any, NamedTuple

import pandas as pd

from .config import load_config

logger = logging.getLogger(__name__)


class Split(NamedTuple):
    """Entraînement et validation. Le test (2015) n'en fait jamais partie."""

    train: pd.DataFrame
    valid: pd.DataFrame


def temporal_split(df: pd.DataFrame, cfg: Mapping[str, Any] | None = None) -> Split:
    """Sépare les prêts nettoyés en entraînement et validation selon la date d'octroi.

    Comportement prévu (M1). L'entraînement garde les prêts dont issue_d est entre
    split.train.start et split.train.end (juin 2007 à décembre 2013), la validation ceux de
    split.valid (2014), bornes incluses. Si une ligne est datée de split.test.start
    (1er janvier 2015) ou après, la fonction lève ValueError : le test ne doit entrer ni dans
    l'entraînement ni dans la validation.

    Corps factice (C2) : les lignes sont triées par date quand la colonne existe, puis les
    premiers 80 % vont à l'entraînement et le reste à la validation. Aucune date n'est contrôlée.

    Paramètres
        df : prêts nettoyés par clean().
        cfg : protocole déjà chargé. Par défaut, load_config().

    Renvoie
        Split(train, valid), deux nouveaux DataFrame. On peut écrire train, valid = temporal_split(df, cfg).
    """
    logger.warning("temporal_split : corps factice (C2), remplacé par M1.")
    cfg = load_config() if cfg is None else cfg
    date_column = cfg["split"]["date_column"]
    ordered = df.sort_values(date_column, kind="stable") if date_column in df.columns else df
    cut = int(len(ordered) * 0.8)
    return Split(train=ordered.iloc[:cut].copy(), valid=ordered.iloc[cut:].copy())

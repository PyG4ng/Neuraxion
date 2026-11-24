"""Étape 3 du pipeline : validation. Implémentation réelle : tâche M2."""

import logging
from collections.abc import Mapping
from typing import Any

from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)

# Clés toujours présentes dans le résultat de evaluate().
METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "brier",
    "precision",
    "recall",
    "acceptance_rate",
    "realized_profit",
)


def evaluate(
    y: ArrayLike,
    proba: ArrayLike,
    decision_threshold: float,
    profit: ArrayLike | None = None,
) -> dict[str, float]:
    """Calcule les métriques de validation d'un modèle.

    Un prêt est refusé quand sa probabilité de défaut est supérieure ou égale à
    decision_threshold, accepté sinon. /predict applique la même règle. La classe positive
    est le défaut (1) : precision et recall portent donc sur les prêts refusés.

    Comportement prévu (M2), une valeur par clé de METRIC_KEYS :
        roc_auc : ROC-AUC, la métrique de promotion ;
        pr_auc : aire sous la courbe précision-rappel (average precision) ;
        brier : score de Brier, plus il est bas, mieux c'est ;
        precision : part de vrais défauts parmi les prêts refusés ;
        recall : part des défauts qui sont refusés ;
        acceptance_rate : part des prêts acceptés ;
        realized_profit : somme de profit sur les prêts acceptés, NaN si profit vaut None.

    Corps factice (C2) : les sept clés, avec des valeurs fixes.

    Paramètres
        y : cible observée, 0 ou 1.
        proba : probabilité de défaut prédite, de même longueur que y.
        decision_threshold : seuil de décision, entre 0 et 1.
        profit : profit réalisé de chaque prêt s'il est accepté (total_pymnt - funded_amnt),
            ou None si ces colonnes manquent.

    Renvoie
        Un dictionnaire dont les clés sont exactement celles de METRIC_KEYS.
    """
    logger.warning("evaluate : corps factice (C2), remplacé par M2.")
    return {
        "roc_auc": 0.5,
        "pr_auc": 0.0,
        "brier": 0.25,
        "precision": 0.0,
        "recall": 0.0,
        "acceptance_rate": 1.0,
        "realized_profit": 0.0 if profit is not None else float("nan"),
    }


def choose_decision_threshold(
    y: ArrayLike,
    proba: ArrayLike,
    profit: ArrayLike,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> float:
    """Choisit le seuil de décision sur la validation.

    Comportement prévu (M2). Pour chaque seuil de la grille decision_threshold.grid
    (0,01 à 0,99 par pas de 0,01), on accepte les prêts dont la probabilité de défaut est
    inférieure au seuil et on additionne leur profit réalisé. On garde le seuil qui donne la
    plus grande somme, et le plus petit en cas d'égalité. y n'entre pas dans le calcul, car le
    profit contient déjà l'issue du prêt : il sert à vérifier que les trois tableaux ont la
    même longueur.

    Corps factice (C2) : renvoie toujours 0,5.

    Paramètres
        y : cible observée sur la validation, 0 ou 1.
        proba : probabilité de défaut prédite sur la validation.
        profit : profit réalisé de chaque prêt s'il est accepté (total_pymnt - funded_amnt).
        cfg : protocole déjà chargé, pour la grille. Par défaut, load_config().

    Renvoie
        Le seuil de décision retenu, entre 0 et 1.
    """
    logger.warning("choose_decision_threshold : corps factice (C2), remplacé par M2.")
    return 0.5

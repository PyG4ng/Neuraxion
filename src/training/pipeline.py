"""Étape 2 du pipeline (training) et enchaînement complet, sans MLflow.

Implémentation réelle : tâche M2.
"""

import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

from .cleaning import clean
from .config import load_config, load_contract, resolve_preset
from .split import temporal_split
from .validation import choose_decision_threshold, evaluate

logger = logging.getLogger(__name__)

# Corps factice : ROC-AUC annoncées par run_pipeline. Avec le seuil provisoire de 0,65,
# default et rapide sont promus et faible est refusé, ce qui permet à P3 de tester les deux cas.
_FAKE_ROC_AUC = {"faible": 0.50}
_FAKE_ROC_AUC_OTHERS = 0.90

# Corps factice : colonnes interdites comme variables (hypothèse H3 du plan), utilisées
# seulement quand aucun contrat n'est disponible.
_FORBIDDEN = ("grade", "sub_grade", "int_rate", "installment", "funded_amnt_inv")


@dataclass(frozen=True)
class PipelineResult:
    """Résultat d'une exécution de run_pipeline(), sans MLflow.

    P3 (train_and_promote) l'enregistre dans MLflow et décide de la promotion avec
    should_promote(metric_value, promotion.threshold, promotion.greater_is_better).

    Attributs
        model : pipeline scikit-learn entraîné. Il prend les colonnes de features, avec les
            valeurs brutes du formulaire, et sa méthode predict_proba donne en colonne 1 la
            probabilité de défaut.
        preset : nom du préréglage (default, rapide, faible), ou name du dictionnaire
            (custom par défaut).
        params : paramètres à enregistrer avec le run, tous de type simple.
        metrics : métriques de validation (2014), avec exactement les clés de METRIC_KEYS.
        metric_name : nom de la métrique de promotion (promotion.metric, donc roc_auc).
        metric_value : valeur de cette métrique sur la validation.
        decision_threshold : seuil de décision choisi sur la validation.
        features : colonnes d'entrée attendues par model, dans l'ordre.
        input_example : quelques lignes d'entrée, pour la signature du modèle dans MLflow.
        contract_version : version du contrat de données utilisé, ou None.
        n_train : nombre de prêts d'entraînement, après l'éventuel tirage.
        n_valid : nombre de prêts de validation.
        duration_s : durée de l'exécution, en secondes.
    """

    model: Pipeline
    preset: str
    params: dict[str, Any]
    metrics: dict[str, float]
    metric_name: str
    metric_value: float
    decision_threshold: float
    features: list[str]
    input_example: pd.DataFrame
    contract_version: str | None
    n_train: int
    n_valid: int
    duration_s: float


def build_pipeline(
    preset: str | Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any] | None = None,
) -> Pipeline:
    """Construit le pipeline scikit-learn, non entraîné, d'un préréglage.

    Comportement prévu (M2). Le pipeline reçoit les colonnes brutes du contrat. Avec V2, une
    première étape ajoute les variables construites (montant sur revenu, solde renouvelable
    sur revenu, ancienneté du crédit, FICO moyen). Un ColumnTransformer construit à partir des
    types du contrat applique ensuite T1 ou T2 aux variables numériques et catégorielles. La
    dernière étape, nommée model, est le modèle du préréglage, avec la graine du protocole.
    Toute la transformation est dans le pipeline : l'entraînement et /predict appliquent les
    mêmes règles (§6 du cahier des charges).

    Corps factice (C2) : un DummyClassifier qui prédit toujours le taux de défaut appris.
    Le préréglage est tout de même contrôlé par resolve_preset().

    Paramètres
        preset : nom de préréglage ou dictionnaire, comme pour resolve_preset().
        contract : contrat de données lu par load_contract().
        cfg : protocole déjà chargé. Par défaut, load_config().

    Renvoie
        Un sklearn.pipeline.Pipeline non entraîné, dont la dernière étape s'appelle model.
    """
    logger.warning("build_pipeline : corps factice (C2), remplacé par M2.")
    resolve_preset(preset, cfg)
    return Pipeline([("model", DummyClassifier(strategy="prior"))])


def run_pipeline(
    df: pd.DataFrame | None = None,
    preset: str | Mapping[str, Any] = "default",
    *,
    cfg: Mapping[str, Any] | None = None,
    contract: Mapping[str, Any] | None = None,
) -> PipelineResult:
    """Enchaîne cleaning, découpage, entraînement et validation, sans MLflow.

    Comportement prévu (M2). clean(), puis temporal_split(). Si le préréglage a un
    sample_size, on tire ce nombre de prêts d'entraînement avec la graine, sans toucher à la
    validation. build_pipeline() est entraîné sur l'entraînement, puis donne les probabilités
    de défaut de la validation 2014. choose_decision_threshold() et evaluate() travaillent
    sur ces probabilités. La fonction n'appelle jamais MLflow : P3 enregistre le résultat.

    Paramètres
        df : prêts bruts (variables du contrat, loan_status, issue_d, funded_amnt et
            total_pymnt). Si None, la fonction lit la table d'entraînement de PostgreSQL avec
            src.data.read_training_frame() : c'est ainsi que train_and_promote(preset)
            l'appelle. Les notebooks passent read_dev_frame(), les tests des données factices.
        preset : nom d'un préréglage (default, rapide, faible), ou dictionnaire des changements
            appliqués à la configuration de référence (voir resolve_preset()).
        cfg : protocole déjà chargé. Par défaut, load_config().
        contract : contrat déjà chargé. Par défaut, load_contract().

    Renvoie
        Un PipelineResult.

    Corps factice (C2). Le vrai enchaînement, avec les corps factices des autres fonctions.
    La ROC-AUC annoncée est fixe : 0,90 pour default, rapide et les dictionnaires, 0,50 pour
    faible. Sans contrat, les colonnes numériques qui ne sont ni la cible, ni la date, ni
    l'évaluation métier, ni interdites servent de variables. Sans df, les prêts viennent de
    make_synthetic_loans(n, seed) si C1 est fusionné, sinon d'un petit jeu interne.
    """
    started = time.perf_counter()
    cfg = load_config() if cfg is None else cfg
    spec = resolve_preset(preset, cfg)
    fake_auc = _FAKE_ROC_AUC.get(spec["name"], _FAKE_ROC_AUC_OTHERS)
    logger.warning(
        "run_pipeline : corps factice (C2), remplacé par M2. ROC-AUC annoncée : %.2f.", fake_auc
    )
    if contract is None:
        contract = _contract_or_empty()
    if df is None:
        df = _fake_loans(cfg["seed"])

    data = clean(df, cfg=cfg)
    train, valid = temporal_split(data, cfg)
    if spec["sample_size"] is not None and len(train) > spec["sample_size"]:
        train = train.sample(n=spec["sample_size"], random_state=cfg["seed"])

    features = _input_columns(contract, spec, data, cfg)
    model = build_pipeline(spec, contract, cfg=cfg)
    model.fit(train[features], _labels(train, cfg))
    proba = _default_probability(model, valid[features])
    y_valid = _labels(valid, cfg)
    profit = _profit(valid, cfg)
    threshold = choose_decision_threshold(y_valid, proba, profit, cfg=cfg)
    metrics = evaluate(y_valid, proba, threshold, profit)
    metrics["roc_auc"] = fake_auc

    metric_name = cfg["promotion"]["metric"]
    version = contract.get("version")
    contract_version = None if version is None else str(version)
    split = cfg["split"]
    params = {
        "preset": spec["name"],
        "model": spec["model"],
        "model_params": json.dumps(spec["model_params"], sort_keys=True),
        "variables": spec["variables"],
        "transformation": spec["transformation"],
        "sample_size": spec["sample_size"],
        "features": ",".join(features),
        "seed": cfg["seed"],
        "protocol_version": str(cfg["protocol_version"]),
        "train_period": f"{split['train']['start']} / {split['train']['end']}",
        "valid_period": f"{split['valid']['start']} / {split['valid']['end']}",
        "n_train": len(train),
        "n_valid": len(valid),
        "decision_threshold": float(threshold),
        "contract_version": contract_version,
    }
    example_source = valid if len(valid) else train
    return PipelineResult(
        model=model,
        preset=spec["name"],
        params=params,
        metrics=metrics,
        metric_name=metric_name,
        metric_value=float(metrics[metric_name]),
        decision_threshold=float(threshold),
        features=features,
        input_example=example_source[features].head(5).copy(),
        contract_version=contract_version,
        n_train=len(train),
        n_valid=len(valid),
        duration_s=time.perf_counter() - started,
    )


def _default_probability(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probabilité de défaut (classe 1) donnée par un modèle entraîné."""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return proba[:, classes.index(1)]


def _labels(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> np.ndarray:
    """Cible de frame. Corps factice : étiquettes alternées si la cible manque ou n'a qu'une classe."""
    column = cfg["target"]["column"]
    if column in frame.columns and frame[column].nunique() == 2:
        return frame[column].to_numpy()
    return np.arange(len(frame)) % 2


def _profit(frame: pd.DataFrame, cfg: Mapping[str, Any]) -> np.ndarray | None:
    """Profit réalisé de chaque prêt s'il est accepté (total_pymnt - funded_amnt), ou None."""
    paid = cfg["business"]["paid_column"]
    amount = cfg["business"]["amount_column"]
    if paid in frame.columns and amount in frame.columns:
        return (frame[paid] - frame[amount]).to_numpy(dtype=float)
    return None


def _input_columns(
    contract: Mapping[str, Any],
    spec: Mapping[str, Any],
    data: pd.DataFrame,
    cfg: Mapping[str, Any],
) -> list[str]:
    """Colonnes d'entrée du modèle : features du préréglage, sinon variables du contrat.

    Corps factice : seules les colonnes présentes dans data sont gardées. Sans contrat, on
    prend les colonnes numériques qui ne sont ni la cible, ni la date, ni l'évaluation métier,
    ni une colonne interdite.
    """
    names = spec["features"] or _contract_feature_names(contract)
    if not names:
        reserved = {
            cfg["target"]["column"],
            cfg["target"]["source"],
            cfg["split"]["date_column"],
            cfg["business"]["amount_column"],
            cfg["business"]["paid_column"],
            *_FORBIDDEN,
        }
        names = [c for c in data.select_dtypes("number").columns if c not in reserved]
    return [c for c in names if c in data.columns]


def _contract_feature_names(contract: Mapping[str, Any]) -> list[str]:
    """Noms des variables du contrat.

    Format supposé en attendant C1 : une clé features, liste d'entrées qui ont chacune un
    name, ou dictionnaire indexé par nom. M2 alignera cette lecture sur le format réel.
    """
    features = contract.get("features") or []
    if isinstance(features, Mapping):
        return [str(name) for name in features]
    return [str(item["name"]) for item in features if isinstance(item, Mapping) and "name" in item]


def _contract_or_empty() -> dict[str, Any]:
    """Corps factice : le contrat de C1 s'il existe déjà, sinon un contrat vide."""
    try:
        return load_contract()
    except FileNotFoundError:
        return {}


def _fake_loans(seed: int) -> pd.DataFrame:
    """Corps factice : prêts utilisés quand aucun DataFrame n'est passé.

    Le vrai run_pipeline lira read_training_frame() (D1) à la place.
    """
    try:
        from ..data.synthetic import make_synthetic_loans
    except ImportError:
        pass
    else:
        return make_synthetic_loans(1000, seed)

    rng = np.random.default_rng(seed)
    n = 200
    amount = rng.integers(1_000, 35_001, n).astype(float)
    months = pd.date_range("2007-06-01", "2014-12-01", freq="MS").to_numpy()
    return pd.DataFrame(
        {
            "loan_amnt": amount,
            "annual_inc": rng.integers(20_000, 150_001, n).astype(float),
            "loan_status": np.resize(["Fully Paid", "Fully Paid", "Charged Off"], n),
            "issue_d": np.resize(months, n),
            "funded_amnt": amount,
            "total_pymnt": amount * rng.uniform(0.3, 1.3, n),
        }
    )

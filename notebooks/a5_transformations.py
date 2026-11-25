"""Transformations T1 et T2 du notebook 05 (tâche A5, lot P4).

Ce module ne sert qu'au notebook 05_comparaison_transformations.ipynb et à ses tests
(tests/notebooks/test_a5_transformations.py). Il est rangé à côté du notebook, et non dans
src/training/, parce que src/training/ appartient au lot P2 : la tâche A5 ne le modifie pas.

Il contient trois choses.

1. make_t1() et make_t2() : les deux ColumnTransformer comparés. T2 reprend T1 et lui ajoute
   des indicateurs de valeurs manquantes, le log1p des montants et le regroupement des
   modalités rares, comme le décrit docs/protocole_validation.md (PR #44).
2. Des remplaçants des fonctions du protocole (clean, temporal_split, evaluate,
   choose_decision_threshold), écrits d'après les docstrings de la PR #44. Le notebook ne les
   utilise que si les vraies fonctions de src.training (M1, M2) manquent ou ont encore leur
   corps factice. Dès que M1 et M2 sont fusionnés, ce sont les vraies qui servent.
3. make_synthetic_loans_a5() : des prêts factices, utilisés seulement quand aucune donnée
   réelle n'est disponible. Les résultats obtenus dessus ne disent rien des vrais prêts.

Tout est ajusté sur l'entraînement seulement : les médianes, les modalités les plus
fréquentes, les moyennes et écarts types, la liste des modalités rares et la liste des
colonnes qui reçoivent un indicateur de manquants. La validation ne fait que traverser le
pipeline déjà ajusté.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------------------
# Protocole v1, recopié de config/training.yaml (PR #44, encore ouverte).
# Sert seulement si load_config() n'est pas disponible. Le notebook compare les deux quand
# le vrai fichier existe et s'arrête s'ils diffèrent sur une valeur qui compte ici.
# ---------------------------------------------------------------------------------------
PROTOCOL_V1: dict[str, Any] = {
    "protocol_version": "v1",
    "seed": 42,
    "scope": {"term_months": 36, "issue_d_start": "2007-06-01", "issue_d_end": "2015-12-31"},
    "target": {
        "column": "target",
        "source": "loan_status",
        "positive": [
            "Charged Off",
            "Default",
            "Does not meet the credit policy. Status:Charged Off",
        ],
        "negative": ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"],
    },
    "split": {
        "date_column": "issue_d",
        "train": {"start": "2007-06-01", "end": "2013-12-31"},
        "valid": {"start": "2014-01-01", "end": "2014-12-31"},
        "test": {"start": "2015-01-01", "end": "2015-12-31"},
    },
    "business": {"amount_column": "funded_amnt", "paid_column": "total_pymnt"},
    "promotion": {"metric": "roc_auc", "greater_is_better": True, "threshold": 0.65},
    "decision_threshold": {
        "method": "max_realized_profit",
        "grid": {"start": 0.01, "stop": 0.99, "step": 0.01},
    },
    "reference": {"model": "logistic_regression", "variables": "V1", "transformation": "T1"},
    "comparison_tags": {
        "axe": ["reference", "modele", "pipeline", "transformation", "seuil", "synthese", "test_final"],
        "lot": ["P1", "P2", "P3", "P4", "P5"],
    },
}

# Clés du protocole dont dépend la comparaison. Si config/training.yaml existe et diffère
# de PROTOCOL_V1 sur l'une d'elles, le notebook le signale.
PROTOCOL_KEYS_USED = (
    "protocol_version",
    "seed",
    "target",
    "split",
    "business",
    "decision_threshold",
    "reference",
)

METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "brier",
    "precision",
    "recall",
    "acceptance_rate",
    "realized_profit",
)

# ---------------------------------------------------------------------------------------
# Variables V1 provisoires, en attendant le contrat v1 (D4, #19).
# Ce sont des colonnes du jeu Lending Club connues au moment de la demande. Aucune colonne
# interdite par le plan (grade, sub_grade, int_rate, installment, funded_amnt_inv, ni rien
# de postérieur à l'octroi) n'y figure. Le notebook prend la liste du contrat dès qu'il existe.
# ---------------------------------------------------------------------------------------
V1_PROVISOIRE: tuple[dict[str, Any], ...] = (
    {"name": "loan_amnt", "type": "numeric", "unit": "dollars"},
    {"name": "annual_inc", "type": "numeric", "unit": "dollars par an"},
    {"name": "dti", "type": "numeric", "unit": "pourcentage"},
    {"name": "revol_bal", "type": "numeric", "unit": "dollars"},
    {"name": "revol_util", "type": "numeric", "unit": "pourcentage"},
    {"name": "fico_range_low", "type": "numeric", "unit": "points"},
    {"name": "open_acc", "type": "numeric", "unit": "comptes"},
    {"name": "total_acc", "type": "numeric", "unit": "comptes"},
    {"name": "delinq_2yrs", "type": "numeric", "unit": "incidents"},
    {"name": "inq_last_6mths", "type": "numeric", "unit": "demandes"},
    {"name": "mort_acc", "type": "numeric", "unit": "comptes"},
    {"name": "emp_length", "type": "categorical", "unit": None},
    {"name": "home_ownership", "type": "categorical", "unit": None},
    {"name": "purpose", "type": "categorical", "unit": None},
    {"name": "verification_status", "type": "categorical", "unit": None},
    {"name": "addr_state", "type": "categorical", "unit": None},
)

# Colonnes interdites comme variables (plan d'équipe, hypothèse H3, reprises par la PR #44).
FORBIDDEN_COLUMNS = ("grade", "sub_grade", "int_rate", "installment", "funded_amnt_inv")

# Montants en dollars : ces variables passent au log1p dans T2. Une variable numérique est un
# montant si son unité est dans AMOUNT_UNITS (casse ignorée) ou si son nom est dans
# AMOUNT_COLUMNS, pour ne pas perdre le log1p si le contrat écrit l'unité autrement.
AMOUNT_UNITS = ("dollars", "dollars par an", "usd", "$", "usd par an")
AMOUNT_COLUMNS = ("loan_amnt", "annual_inc", "revol_bal", "tot_cur_bal", "total_rev_hi_lim", "avg_cur_bal")

# Part minimale d'une modalité dans l'entraînement pour garder sa propre colonne dans T2.
# En dessous, la modalité rejoint la colonne commune des modalités rares.
RARE_MIN_FREQUENCY = 0.01


@dataclass(frozen=True)
class FeatureSpec:
    """Variables d'entrée, rangées selon la transformation qu'elles reçoivent."""

    numeric: list[str]
    categorical: list[str]
    amounts: list[str]
    source: str  # "contrat <version>" ou "V1 provisoire (A5)"

    @property
    def all(self) -> list[str]:
        return [*self.numeric, *self.categorical]


def feature_spec(contract: Mapping[str, Any] | None) -> FeatureSpec:
    """Range les variables du contrat (ou de V1_PROVISOIRE) en numériques, catégorielles et montants.

    Format de contrat lu : une clé features, liste d'entrées {name, type, unit, ...}, avec
    type numeric ou categorical. C'est le format du contrat d'illustration de docs/api.md
    (lot P4) et celui que suppose la PR #44. Si D4 choisit un autre format, seule cette
    fonction change.

    Un montant est une variable numérique dont l'unité est dans AMOUNT_UNITS ou dont le nom
    est dans AMOUNT_COLUMNS.
    """
    entries: Sequence[Mapping[str, Any]]
    if contract and contract.get("features"):
        raw = contract["features"]
        if isinstance(raw, Mapping):
            entries = [{"name": name, **(value or {})} for name, value in raw.items()]
        else:
            entries = list(raw)
        source = f"contrat {contract.get('version', '?')}"
    else:
        entries = V1_PROVISOIRE
        source = "V1 provisoire (A5)"

    numeric, categorical, amounts = [], [], []
    for entry in entries:
        name = str(entry["name"])
        if name in FORBIDDEN_COLUMNS:
            raise ValueError(f"{name} est une colonne interdite : elle ne peut pas être une variable.")
        kind = entry.get("type")
        if kind in ("numeric", "integer", "number", "float"):
            numeric.append(name)
            unit = str(entry.get("unit") or "").strip().casefold()
            if unit in AMOUNT_UNITS or name in AMOUNT_COLUMNS:
                amounts.append(name)
        elif kind in ("categorical", "category", "string"):
            categorical.append(name)
        else:
            raise ValueError(f"Type inconnu pour {name} : {kind!r}.")
    return FeatureSpec(numeric=numeric, categorical=categorical, amounts=amounts, source=source)


# ---------------------------------------------------------------------------------------
# T1 et T2
# ---------------------------------------------------------------------------------------
def _onehot(min_frequency: float | None) -> OneHotEncoder:
    if min_frequency is None:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    # Les modalités rares de l'entraînement et les modalités jamais vues partagent la
    # colonne commune infrequent_sklearn.
    return OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        sparse_output=False,
    )


def _numeric_t1() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])


def _categorical(min_frequency: float | None) -> Pipeline:
    return Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", _onehot(min_frequency))]
    )


def make_t1(spec: FeatureSpec) -> ColumnTransformer:
    """T1 : médiane (numériques) ou modalité la plus fréquente (catégorielles), standardisation, one-hot.

    Une modalité jamais vue à l'entraînement donne une ligne de zéros dans le one-hot.
    """
    return ColumnTransformer(
        [
            ("numeriques", _numeric_t1(), list(spec.numeric)),
            ("categorielles", _categorical(None), list(spec.categorical)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_t2(spec: FeatureSpec, min_frequency: float = RARE_MIN_FREQUENCY) -> ColumnTransformer:
    """T2 : T1, plus indicateurs de manquants, log1p des montants et regroupement des modalités rares.

    - Indicateurs : une colonne 0/1 par variable qui a des manquants dans l'entraînement
      (MissingIndicator, features="missing-only"). Une variable sans manquant à
      l'entraînement n'en reçoit pas, même si la validation en a.
    - Montants (spec.amounts) : log1p avant la médiane et la standardisation. Un montant
      négatif donnerait un log non défini : il devient manquant, puis reçoit la médiane.
      En pratique clean() a déjà mis à manquant les valeurs impossibles.
    - Modalités rares : moins de min_frequency (1 %) des prêts d'entraînement, regroupées
      dans une seule colonne, qui reçoit aussi les modalités jamais vues.
    """
    amounts = [c for c in spec.numeric if c in spec.amounts]
    others = [c for c in spec.numeric if c not in spec.amounts]
    log_amounts = Pipeline(
        [
            ("log1p", FunctionTransformer(_log1p_nonnegative, feature_names_out="one-to-one")),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("montants", log_amounts, amounts),
            ("numeriques", _numeric_t1(), others),
            ("categorielles", _categorical(min_frequency), list(spec.categorical)),
            (
                "manquants",
                MissingIndicator(features="missing-only", error_on_new=False),
                spec.all,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def _log1p_nonnegative(x: Any) -> np.ndarray:
    """log1p des montants ; une valeur négative devient manquante (NaN)."""
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, np.nan)
    ok = arr >= 0
    out[ok] = np.log1p(arr[ok])
    return out


def to_sklearn_types(X: pd.DataFrame) -> pd.DataFrame:
    """Types simples pour scikit-learn : float64 pour les numériques, object pour le reste.

    clean() (M1) peut produire des types nullables de pandas (Int64, Float64, string) avec
    pd.NA, que MissingIndicator et SimpleImputer ne savent pas traiter. Les manquants
    deviennent np.nan dans tous les cas. Première étape du pipeline : /predict l'applique aussi.
    """
    out = {}
    for col in X.columns:
        series = X[col]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            out[col] = pd.to_numeric(series, errors="coerce").astype("float64")
        else:
            values = series.astype(object)
            out[col] = values.where(values.notna(), np.nan)
    return pd.DataFrame(out, index=X.index)


TRANSFORMATIONS: dict[str, Callable[[FeatureSpec], ColumnTransformer]] = {
    "T1": make_t1,
    "T2": make_t2,
}


def make_pipeline(transformation: str, spec: FeatureSpec, seed: int) -> Pipeline:
    """Pipeline complet de la comparaison : types, transformation, puis régression logistique.

    Le modèle est le même pour T1 et T2 : seule la transformation change. max_iter vaut
    2000 pour laisser lbfgs converger ; un avertissement de non-convergence est relevé par le
    notebook et compté comme un échec partiel.
    """
    return Pipeline(
        [
            ("types", FunctionTransformer(to_sklearn_types, feature_names_out="one-to-one")),
            ("transformation", TRANSFORMATIONS[transformation](spec)),
            ("model", LogisticRegression(max_iter=2000, random_state=seed)),
        ]
    )


# ---------------------------------------------------------------------------------------
# Remplaçants des fonctions du protocole (src.training, PR #44, M1 et M2)
# ---------------------------------------------------------------------------------------
def clean_local(df: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Version minimale de clean() (M1) : cible, périmètre, dates, doublons.

    Ne fait pas ce que M1 doit ajouter (valeurs impossibles, variables dérivées, types du
    contrat). Le DataFrame reçu n'est pas modifié.
    """
    target = cfg["target"]
    out = df.copy()
    mapping = {s: 1 for s in target["positive"]} | {s: 0 for s in target["negative"]}
    out[target["column"]] = out[target["source"]].map(mapping)
    out = out.dropna(subset=[target["column"]])
    out[target["column"]] = out[target["column"]].astype("int64")
    date = cfg["split"]["date_column"]
    out[date] = pd.to_datetime(out[date])
    scope = cfg["scope"]
    out = out[(out[date] >= scope["issue_d_start"]) & (out[date] <= scope["issue_d_end"])]
    if "term" in out.columns:
        months = out["term"].astype(str).str.extract(r"(\d+)")[0].astype(float)
        out = out[months == scope["term_months"]]
    return out.drop_duplicates().reset_index(drop=True)


def temporal_split_local(df: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage du protocole : entraînement 2007-06 à 2013-12, validation 2014, bornes incluses.

    Lève ValueError dès qu'une ligne est datée du début du test (2015-01-01) ou après :
    le test ne doit jamais arriver jusqu'ici.
    """
    split = cfg["split"]
    # Les bornes sont des jours : un prêt du 31/12/2013 à 12 h reste dans l'entraînement.
    date = pd.to_datetime(df[split["date_column"]]).dt.normalize()
    test_start = pd.Timestamp(split["test"]["start"])
    if (date >= test_start).any():
        raise ValueError(
            f"{int((date >= test_start).sum())} prêts sont datés de {test_start.date()} ou après : "
            "le test ne doit pas être lu dans ce notebook."
        )
    in_train = date.between(split["train"]["start"], split["train"]["end"])
    in_valid = date.between(split["valid"]["start"], split["valid"]["end"])
    return df[in_train].copy(), df[in_valid].copy()


def evaluate_local(
    y: Any, proba: Any, decision_threshold: float, profit: Any | None = None
) -> dict[str, float]:
    """Les sept métriques du protocole. Refus si proba >= decision_threshold."""
    y = np.asarray(y)
    proba = np.asarray(proba, dtype=float)
    reject = proba >= decision_threshold
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "precision": float(precision_score(y, reject, zero_division=0)),
        "recall": float(recall_score(y, reject, zero_division=0)),
        "acceptance_rate": float(np.mean(~reject)),
        "realized_profit": (
            float(np.sum(np.asarray(profit, dtype=float)[~reject])) if profit is not None else float("nan")
        ),
    }


def threshold_grid(cfg: Mapping[str, Any]) -> np.ndarray:
    grid = cfg["decision_threshold"]["grid"]
    n = int(round((grid["stop"] - grid["start"]) / grid["step"])) + 1
    return np.round(grid["start"] + grid["step"] * np.arange(n), 10)


def choose_threshold_local(y: Any, proba: Any, profit: Any, cfg: Mapping[str, Any]) -> float:
    """Seuil de la grille qui maximise le profit réalisé des prêts acceptés (le plus petit si égalité)."""
    proba = np.asarray(proba, dtype=float)
    profit = np.asarray(profit, dtype=float)
    if not (len(np.asarray(y)) == len(proba) == len(profit)):
        raise ValueError("y, proba et profit doivent avoir la même longueur.")
    grid = threshold_grid(cfg)
    totals = np.array([profit[proba < t].sum() for t in grid])
    return float(grid[int(np.argmax(totals))])


def paired_bootstrap_auc(
    y: Any, proba_a: Any, proba_b: Any, n_boot: int = 300, seed: int = 42
) -> dict[str, float]:
    """Écart de ROC-AUC (b - a) sur la validation et son intervalle bootstrap à 95 %.

    Les mêmes prêts sont tirés pour les deux variantes (bootstrap apparié). Ce n'est pas un
    élément du protocole : c'est un ajout du notebook 05 pour juger si l'écart dépasse le
    bruit d'échantillonnage de la validation.
    """
    y = np.asarray(y)
    a = np.asarray(proba_a, dtype=float)
    b = np.asarray(proba_b, dtype=float)
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    diffs = np.asarray(diffs)
    return {
        "diff": float(roc_auc_score(y, b) - roc_auc_score(y, a)),
        "low": float(np.quantile(diffs, 0.025)),
        "high": float(np.quantile(diffs, 0.975)),
        "n_boot": int(len(diffs)),
    }


# ---------------------------------------------------------------------------------------
# Détection des corps factices de la PR #44
# ---------------------------------------------------------------------------------------
class _Collector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def probe_status(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
    """État d'une fonction de src.training, d'après un appel d'essai.

    Renvoie "factice" si l'appel écrit l'avertissement « corps factice » des corps provisoires
    de la PR #44, "reelle" sinon, ou "erreur : <type> : <message>" si l'appel échoue. Une
    fonction réelle (M1, M2) n'écrit plus cet avertissement.
    """
    collector = _Collector()
    root = logging.getLogger("src")
    # L'avertissement doit arriver même si les logs sont coupés ou réglés plus haut.
    previous_disable = logging.root.manager.disable
    previous_level = root.level
    logging.disable(logging.NOTSET)
    root.setLevel(logging.WARNING)
    root.addHandler(collector)
    try:
        fn(*args, **kwargs)
    except Exception as exc:  # l'appelant décide : le notebook le note dans ses échecs
        return f"erreur : {type(exc).__name__} : {exc}"
    finally:
        root.removeHandler(collector)
        root.setLevel(previous_level)
        logging.disable(previous_disable)
    return "factice" if any("corps factice" in m for m in collector.messages) else "reelle"


def is_stub(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """Vrai si fn n'est pas utilisable telle quelle : corps factice, ou échec de l'appel d'essai."""
    return probe_status(fn, *args, **kwargs) != "reelle"


# ---------------------------------------------------------------------------------------
# Prêts factices
# ---------------------------------------------------------------------------------------
def make_synthetic_loans_a5(n: int = 40_000, seed: int = 42) -> pd.DataFrame:
    """Prêts factices au format brut (variables V1 provisoires, loan_status, issue_d,
    funded_amnt, total_pymnt), de juin 2007 à décembre 2014, jamais 2015.

    Ils reproduisent quelques traits qui comptent pour T1 contre T2 : montants très
    asymétriques, manquants informatifs (emp_length, revol_util), une variable qui n'existe
    pas au début de la période (mort_acc avant 2010), des modalités rares (addr_state,
    purpose), et un taux de défaut qui varie d'une année à l'autre.

    Le lien entre variables et défaut est choisi ici, par ce générateur. Il ne dit rien des
    vrais prêts : un écart entre T1 et T2 sur ces données ne reflète que ces choix.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2007-06-01", "2014-12-01", freq="MS")
    # Plus de prêts vers la fin, comme dans le vrai jeu.
    weights = np.linspace(1, 12, len(months))
    issue = rng.choice(months.to_numpy(), size=n, p=weights / weights.sum())
    year = pd.DatetimeIndex(issue).year.to_numpy()

    annual_inc = np.round(np.exp(rng.normal(11.0, 0.55, n)), -2)
    loan_amnt = np.clip(np.round(np.exp(rng.normal(9.3, 0.6, n)), -2), 1000, 35000)
    revol_bal = np.round(np.exp(rng.normal(9.2, 1.1, n)))
    dti = np.clip(rng.normal(17, 8, n), 0, 45)
    revol_util = np.clip(rng.normal(53, 24, n), 0, 130)
    fico = np.clip(np.round(rng.normal(695, 30, n) / 5) * 5, 660, 845)
    open_acc = rng.poisson(10, n) + 1
    total_acc = open_acc + rng.poisson(13, n)
    delinq = rng.poisson(0.3, n)
    inq = rng.poisson(0.9, n)
    mort_acc = rng.poisson(1.6, n).astype(float)

    emp_levels = np.array(["< 1 year", "1 year", "2 years", "3 years", "5 years", "10+ years"])
    emp = rng.choice(emp_levels, size=n, p=[0.1, 0.08, 0.1, 0.1, 0.22, 0.4]).astype(object)
    home = rng.choice(["MORTGAGE", "RENT", "OWN", "OTHER", "NONE"], size=n, p=[0.48, 0.41, 0.1, 0.007, 0.003])
    purpose = rng.choice(
        ["debt_consolidation", "credit_card", "home_improvement", "other", "major_purchase",
         "small_business", "wedding", "renewable_energy", "educational"],
        size=n,
        p=[0.58, 0.21, 0.06, 0.07, 0.04, 0.025, 0.01, 0.003, 0.002],
    )
    verif = rng.choice(["Not Verified", "Source Verified", "Verified"], size=n, p=[0.35, 0.35, 0.3])
    common = ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "VA"]
    rare = ["ND", "ME", "IA", "NE", "ID", "SD", "WY", "VT", "AK", "MS"]
    state_p = np.r_[np.full(len(common), 0.095), np.full(len(rare), 0.005)]
    state = rng.choice(common + rare, size=n, p=state_p / state_p.sum())

    # Risque latent : choisi par le générateur, avec le revenu en log.
    logit = (
        -2.0
        - 0.55 * (np.log(annual_inc) - 11.0)
        + 0.30 * (np.log(loan_amnt) - 9.3)
        + 0.035 * (dti - 17)
        - 0.018 * (fico - 695)
        + 0.006 * (revol_util - 53)
        + 0.12 * inq
        + 0.25 * (purpose == "small_business")
        + 0.15 * (home == "RENT")
        + 0.10 * (year - 2011) / 3
    )
    emp_missing = rng.random(n) < 0.05 + 0.04 * (logit > -1.6)  # manquant informatif
    emp[emp_missing] = np.nan
    logit = logit + 0.3 * emp_missing
    revol_util[rng.random(n) < 0.02] = np.nan
    mort_acc[year < 2010] = np.nan
    default = rng.random(n) < 1 / (1 + np.exp(-logit))

    funded = loan_amnt.copy()
    paid_ratio = np.where(default, rng.uniform(0.2, 0.9, n), rng.uniform(1.05, 1.25, n))
    status = np.where(default, "Charged Off", "Fully Paid").astype(object)
    flip = rng.random(n) < 0.01  # quelques statuts « credit policy » du vrai jeu
    status[flip & default] = "Does not meet the credit policy. Status:Charged Off"
    status[flip & ~default] = "Does not meet the credit policy. Status:Fully Paid"

    return pd.DataFrame(
        {
            "loan_amnt": loan_amnt,
            "annual_inc": annual_inc,
            "dti": np.round(dti, 2),
            "revol_bal": revol_bal,
            "revol_util": np.round(revol_util, 1),
            "fico_range_low": fico,
            "open_acc": open_acc.astype(float),
            "total_acc": total_acc.astype(float),
            "delinq_2yrs": delinq.astype(float),
            "inq_last_6mths": inq.astype(float),
            "mort_acc": mort_acc,
            "emp_length": emp,
            "home_ownership": home,
            "purpose": purpose,
            "verification_status": verif,
            "addr_state": state,
            "term": " 36 months",
            "loan_status": status,
            "issue_d": issue,
            "funded_amnt": funded,
            "total_pymnt": np.round(funded * paid_ratio, 2),
        }
    )

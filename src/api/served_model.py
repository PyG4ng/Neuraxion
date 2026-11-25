"""Modèle servi par POST /predict : adaptateur autour de load_champion() (tâche A2).

load_champion() est livré par la tâche R1 (lot P3) dans src/registry/champion.py. Ce module
est le seul point de contact avec R1 :

- il importe src.registry.champion au moment du chargement, pas à l'import de l'API ;
- un module absent (R1 pas encore livré) compte comme « pas de champion », avec un
  avertissement clair dans les logs ;
- il contrôle ce que load_champion() renvoie et le transforme en ServedChampion.

Interface attendue de load_champion(), appelée sans argument (issue #8) : None s'il n'y a pas
de champion, sinon un objet (attributs) ou un dictionnaire (clés) qui fournit :

- model : le modèle chargé, avec predict_proba (par exemple mlflow.sklearn.load_model) ;
- version : la version résolue depuis l'alias, "3" ou 3 ;
- decision_threshold : le seuil de décision, entre 0 et 1 (tag decision_threshold) ;
- features (ou feature_names) : la liste des variables attendues par le modèle ;
- name (ou model_name), facultatif : sinon MLFLOW_MODEL_NAME, puis credit-default.

Les variables du modèle doivent être exactement celles du contrat de données ; l'ordre est
celui du modèle.
"""

import importlib
import inspect
import logging
import math
import numbers
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.api.schemas import MODEL_VERSION_PATTERN, ModelInfo

logger = logging.getLogger("neuraxion.api")

CHAMPION_MODULE = "src.registry.champion"
DEFAULT_MODEL_NAME = "credit-default"


class ChampionError(ValueError):
    """Ce que load_champion() renvoie ne peut pas être servi. Le message ne cite aucune donnée."""


class ModelOutputError(ValueError):
    """Le modèle a renvoyé autre chose qu'une probabilité de défaut entre 0 et 1.

    Levée seulement par ce module, avec l'un des textes fixes de OUTPUT_PROBLEMS.
    """


class ModelFailure(RuntimeError):
    """predict_proba a levé une exception. Seul le type de cette exception est gardé."""

    def __init__(self, error_type: str) -> None:
        super().__init__(error_type)
        self.error_type = error_type


OUTPUT_SHAPE = "predict_proba doit renvoyer une ligne par demande et deux colonnes (classes 0 et 1)."
OUTPUT_NOT_REAL = "La probabilité renvoyée par le modèle n'est pas un nombre réel."
OUTPUT_RANGE = "La probabilité renvoyée par le modèle n'est pas entre 0 et 1."
OUTPUT_PROBLEMS = (OUTPUT_SHAPE, OUTPUT_NOT_REAL, OUTPUT_RANGE)


@dataclass(frozen=True)
class ServedChampion:
    """Modèle chargé une fois et gardé en mémoire, avec ce qui décrit sa décision."""

    model: Any
    name: str
    version: str
    decision_threshold: float
    features: tuple[str, ...]
    categorical: frozenset[str]
    positive_column: int

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, version=self.version, decision_threshold=self.decision_threshold)

    def frame(self, input_data: Mapping[str, Any]) -> pd.DataFrame:
        """Une ligne, colonnes dans l'ordre du modèle. Valeur absente : NaN.

        Les variables catégorielles sont de type object, les autres float64, que la valeur
        soit renseignée ou non : le type d'une colonne ne dépend pas de la demande.
        """
        columns = {}
        for name in self.features:
            value = input_data.get(name)
            if name in self.categorical:
                columns[name] = pd.Series([np.nan if value is None else value], dtype=object)
            else:
                columns[name] = pd.Series([np.nan if value is None else value], dtype="float64")
        return pd.DataFrame(columns)

    def default_probability(self, input_data: Mapping[str, Any]) -> float:
        """Probabilité de défaut (classe 1).

        Lève ModelFailure si predict_proba lève une exception (quelle qu'elle soit, même une
        ModelOutputError), ModelOutputError si sa sortie est inexploitable.
        """
        frame = self.frame(input_data)
        try:
            raw = self.model.predict_proba(frame)
        except Exception as error:  # noqa: BLE001 - seul le type est gardé, jamais le message
            raise ModelFailure(type(error).__name__) from None
        try:
            scores = np.asarray(raw)
        except Exception:  # noqa: BLE001 - sortie que numpy ne sait pas lire
            raise ModelOutputError(OUTPUT_NOT_REAL) from None
        if scores.shape != (1, 2):
            raise ModelOutputError(OUTPUT_SHAPE)
        # Réels seulement : un complexe perdrait sa partie imaginaire, un booléen n'est pas une probabilité.
        if scores.dtype.kind not in "fiu":
            raise ModelOutputError(OUTPUT_NOT_REAL)
        value = float(scores[0, self.positive_column])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ModelOutputError(OUTPUT_RANGE)
        return value

    def decide(self, probability: float) -> str:
        """reject si la probabilité atteint le seuil, sinon approve (docs/api.md)."""
        return "reject" if probability >= self.decision_threshold else "approve"


def load_champion_from_registry() -> Any:
    """Appelle load_champion() de R1. None si le module n'existe pas encore."""
    try:
        module = importlib.import_module(CHAMPION_MODULE)
    except ModuleNotFoundError as error:
        # Seule l'absence du module lui-même vaut « pas de champion » ; une dépendance absente
        # à l'intérieur de R1 (mlflow, par exemple) reste une erreur de chargement.
        if error.name not in (CHAMPION_MODULE, "src.registry"):
            raise
        logger.warning(
            "Module %s introuvable (tâche R1 non livrée) : aucun modèle n'est servi, "
            "POST /predict répond 503.",
            CHAMPION_MODULE,
        )
        return None
    return module.load_champion()


def _read(champion: Any, *names: str) -> Any:
    """Premier attribut (ou première clé) renseigné parmi names, sinon None."""
    for name in names:
        value = champion.get(name) if isinstance(champion, Mapping) else getattr(champion, name, None)
        if value is not None:
            return value
    return None


def _version(value: Any) -> str:
    if isinstance(value, bool | np.bool_) or not isinstance(value, str | numbers.Integral):
        raise ChampionError("version doit être un entier ou un texte (\"3\").")
    text = str(int(value)) if isinstance(value, numbers.Integral) else value.strip()
    if not re.fullmatch(MODEL_VERSION_PATTERN, text):
        raise ChampionError("version doit être un entier positif (\"3\"), jamais l'alias.")
    return text


def _threshold(value: Any) -> float:
    if isinstance(value, bool | np.bool_):
        raise ChampionError("decision_threshold doit être un nombre entre 0 et 1.")
    try:
        threshold = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ChampionError("decision_threshold doit être un nombre entre 0 et 1.") from error
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ChampionError("decision_threshold doit être un nombre entre 0 et 1.")
    return threshold


def _features(value: Any, expected: Iterable[str]) -> tuple[str, ...]:
    # Une séquence ordonnée : un dictionnaire ou un ensemble ne fixe pas l'ordre des colonnes.
    if not isinstance(value, list | tuple | np.ndarray | pd.Index) or getattr(value, "ndim", 1) != 1:
        raise ChampionError("features doit être une liste de noms de variables.")
    features = tuple(value)
    if not features or not all(isinstance(name, str) for name in features):
        raise ChampionError("features doit être une liste non vide de noms de variables.")
    if len(set(features)) != len(features):
        raise ChampionError("features contient deux fois la même variable.")
    expected = set(expected)
    missing, extra = sorted(expected - set(features)), sorted(set(features) - expected)
    if missing or extra:
        # Ces noms viennent du contrat et du registre, jamais d'une demande.
        parts = []
        if missing:
            parts.append(f"absentes du modèle : {', '.join(missing)}")
        if extra:
            parts.append(f"absentes du contrat : {', '.join(extra)}")
        raise ChampionError("Variables du modèle différentes du contrat de données (" + " ; ".join(parts) + ").")
    return features


def _positive_column(model: Any) -> int:
    """Colonne de predict_proba qui correspond à la classe 1 (défaut)."""
    if not callable(getattr(model, "predict_proba", None)):
        raise ChampionError("Le modèle doit exposer predict_proba (par exemple mlflow.sklearn.load_model).")
    classes = getattr(model, "classes_", None)
    if classes is None:
        # Sans classes_, rien ne dit quelle colonne est le défaut : une probabilité inversée
        # passerait sans alerte.
        raise ChampionError("Le modèle doit exposer classes_, pour situer la classe 1 (défaut) dans predict_proba.")
    classes = np.asarray(classes)
    # Exactement deux classes, 0 et 1, sur une dimension : avec une troisième, la colonne de la
    # classe 1 ne serait plus la probabilité de défaut.
    labels = list(classes) if classes.ndim == 1 else []
    numbers_only = all(isinstance(label, numbers.Real | np.bool_) for label in labels)
    if len(labels) != 2 or not numbers_only or sorted(labels) != [0, 1]:
        raise ChampionError("classes_ du modèle doit contenir exactement 0 et 1 (1 pour le défaut).")
    return 0 if labels[0] == 1 else 1


def to_served_model(champion: Any, expected_features: Iterable[str], categorical: Iterable[str]) -> ServedChampion:
    """Contrôle ce que load_champion() renvoie. Lève ChampionError s'il ne peut pas être servi."""
    model = _read(champion, "model")
    if model is None:
        raise ChampionError("load_champion() n'a pas fourni de modèle (attribut model).")
    name = _read(champion, "name", "model_name") or os.environ.get("MLFLOW_MODEL_NAME") or DEFAULT_MODEL_NAME
    if not isinstance(name, str) or not name.strip():
        raise ChampionError("Le nom du modèle doit être un texte non vide.")
    served = ServedChampion(
        model=model,
        name=name.strip(),
        version=_version(_read(champion, "version", "model_version")),
        decision_threshold=_threshold(_read(champion, "decision_threshold")),
        features=_features(_read(champion, "features", "feature_names"), expected_features),
        categorical=frozenset(categorical),
        positive_column=_positive_column(model),
    )
    try:
        served.info  # noqa: B018 - même contrôle que /health
    except ValidationError as error:
        raise ChampionError("Description du modèle invalide (nom, version ou seuil).") from error
    return served


def adapt_champion(
    champion: Any, expected_features: Iterable[str], categorical: Iterable[str]
) -> ServedChampion | None:
    """Transforme ce que load_champion() a renvoyé. None s'il n'y a pas de champion.

    Lève ChampionError si ce résultat ne peut pas être servi, quelle que soit la cause
    (contrôle de l'interface ou erreur imprévue pendant l'adaptation, dont seul le type est
    gardé). Les exceptions levées par load_champion() lui-même ne passent pas par ici.
    """
    if inspect.isawaitable(champion):
        if inspect.iscoroutine(champion):
            champion.close()
        raise ChampionError("load_champion() doit être une fonction synchrone.")
    if champion is None:
        return None
    try:
        return to_served_model(champion, expected_features, categorical)
    except ChampionError:
        raise
    except Exception as error:  # noqa: BLE001 - résultat de R1 inexploitable : seul le type est gardé
        raise ChampionError(f"Résultat de load_champion() inexploitable ({type(error).__name__}).") from None

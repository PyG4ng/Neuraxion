"""Lecture du protocole (config/training.yaml) et du contrat de données.

Ces fonctions sont réelles dès le contrat C2 : les notebooks et les autres lots peuvent
s'en servir tout de suite.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

# Racine du dépôt : src/training/config.py, puis src/training, puis src, puis la racine.
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "training.yaml"
CONTRACT_PATH = REPO_ROOT / "contracts" / "features.yaml"

PRESET_NAMES = ("default", "rapide", "faible")
MODELS = ("logistic_regression", "hist_gradient_boosting", "random_forest")
VARIABLE_SETS = ("V1", "V2")
TRANSFORMATIONS = ("T1", "T2")

_REQUIRED_KEYS = (
    "protocol_version",
    "seed",
    "scope",
    "target",
    "split",
    "business",
    "promotion",
    "decision_threshold",
    "reference",
    "presets",
    "comparison_tags",
)


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Lit le protocole d'entraînement et vérifie qu'il est complet.

    Paramètres
        path : chemin du fichier YAML. Par défaut, config/training.yaml à la racine du dépôt.

    Renvoie
        Le contenu du fichier, sous forme de dictionnaire.

    Lève
        FileNotFoundError si le fichier n'existe pas.
        ValueError s'il manque une clé obligatoire, une partie du découpage ou un préréglage.
    """
    path = CONFIG_PATH if path is None else Path(path)
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} ne contient pas un dictionnaire YAML.")
    missing = [key for key in _REQUIRED_KEYS if key not in cfg]
    split = cfg.get("split") or {}
    presets = cfg.get("presets") or {}
    missing += [f"split.{part}" for part in ("train", "valid", "test") if part not in split]
    missing += [f"presets.{name}" for name in PRESET_NAMES if name not in presets]
    if missing:
        raise ValueError(f"{path} : clés manquantes : {', '.join(missing)}.")
    return cfg


def load_contract(path: str | Path | None = None) -> dict[str, Any]:
    """Lit le contrat de données (contracts/features.yaml, tenu par P1).

    Le contenu est renvoyé tel quel : chaque module y prend ce dont il a besoin.

    Paramètres
        path : chemin du contrat. Par défaut, contracts/features.yaml à la racine du dépôt.

    Lève
        FileNotFoundError si le contrat n'existe pas encore.
        ValueError si le fichier ne contient pas un dictionnaire YAML.
    """
    path = CONTRACT_PATH if path is None else Path(path)
    with path.open(encoding="utf-8") as f:
        contract = yaml.safe_load(f)
    if not isinstance(contract, dict):
        raise ValueError(f"{path} ne contient pas un dictionnaire YAML.")
    return contract


def resolve_preset(
    preset: str | Mapping[str, Any], cfg: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Renvoie la configuration complète d'un préréglage.

    Paramètres
        preset : nom d'un préréglage de POST /train (default, rapide ou faible), ou
            dictionnaire des seuls changements à appliquer à la configuration de référence
            (section reference), par exemple {"model": "hist_gradient_boosting"} dans un
            notebook de comparaison. Un dictionnaire peut porter son propre name.
        cfg : protocole déjà chargé. Par défaut, load_config().

    Renvoie
        Un nouveau dictionnaire avec les clés name, model, model_params, variables,
        transformation, sample_size (None pour toute la période) et features (None pour
        toutes les variables du contrat).

    Lève
        ValueError pour un préréglage inconnu ou une valeur non permise.
    """
    cfg = load_config() if cfg is None else cfg
    if isinstance(preset, str):
        if preset not in cfg["presets"]:
            allowed = ", ".join(cfg["presets"])
            raise ValueError(f"Préréglage inconnu : {preset}. Valeurs permises : {allowed}.")
        spec = {"name": preset, **cfg["presets"][preset]}
    elif isinstance(preset, Mapping):
        spec = {**cfg["reference"], **preset}
        spec.setdefault("name", "custom")
    else:
        raise ValueError("preset doit être un nom de préréglage ou un dictionnaire.")

    spec["model_params"] = dict(spec.get("model_params") or {})
    spec.setdefault("sample_size", None)
    spec.setdefault("features", None)
    _check_choice("model", spec.get("model"), MODELS)
    _check_choice("variables", spec.get("variables"), VARIABLE_SETS)
    _check_choice("transformation", spec.get("transformation"), TRANSFORMATIONS)

    size = spec["sample_size"]
    if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size <= 0):
        raise ValueError(f"sample_size doit être un entier positif ou null, pas {size!r}.")
    features = spec["features"]
    if features is not None:
        if isinstance(features, str) or not all(isinstance(name, str) for name in features):
            raise ValueError("features doit être une liste de noms de colonnes, ou null.")
        spec["features"] = list(features)
    return spec


def comparison_tags(
    axe: str, variante: str, lot: str, cfg: Mapping[str, Any] | None = None
) -> dict[str, str]:
    """Construit les quatre tags d'un run de comparaison (expérience credit-exploration).

    Paramètres
        axe : facteur comparé. Valeurs permises dans comparison_tags.axe du protocole
            (reference, modele, pipeline, transformation, seuil, synthese, test_final).
        variante : nom court de la variante, par exemple logreg, hgb, V2 ou T2.
        lot : lot qui lance le run (P1 à P5).
        cfg : protocole déjà chargé. Par défaut, load_config().

    Renvoie
        {"axe": ..., "variante": ..., "lot": ..., "protocole": ...}, à passer tel quel à MLflow.

    Lève
        ValueError si l'axe ou le lot ne sont pas permis, ou si la variante est vide.
    """
    cfg = load_config() if cfg is None else cfg
    allowed = cfg["comparison_tags"]
    _check_choice("axe", axe, tuple(allowed["axe"]))
    _check_choice("lot", lot, tuple(allowed["lot"]))
    if not isinstance(variante, str) or not variante.strip():
        raise ValueError("variante doit être un texte court non vide, par exemple hgb ou T2.")
    return {
        "axe": axe,
        "variante": variante.strip(),
        "lot": lot,
        "protocole": str(cfg["protocol_version"]),
    }


def _check_choice(key: str, value: Any, allowed: tuple[str, ...]) -> None:
    """Lève ValueError si value ne fait pas partie des valeurs permises pour key."""
    if value not in allowed:
        raise ValueError(f"{key} = {value!r} n'est pas permis. Valeurs permises : {', '.join(allowed)}.")

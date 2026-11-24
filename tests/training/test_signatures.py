"""Forme du contrat C2.

Ces tests vérifient les signatures et la forme des résultats, pas les valeurs des corps
factices : ils restent valables quand M1 et M2 remplacent ces corps.
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.training import (
    METRIC_KEYS,
    PipelineResult,
    choose_decision_threshold,
    comparison_tags,
    evaluate,
    load_config,
    resolve_preset,
    run_pipeline,
)


@pytest.fixture
def cfg() -> dict:
    return load_config()


@pytest.fixture
def validation_a_la_main() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Six prêts de validation : cible, probabilité de défaut et profit réalisé."""
    y = np.array([0, 1, 0, 1, 0, 0])
    proba = np.array([0.1, 0.8, 0.3, 0.6, 0.2, 0.4])
    profit = np.array([500.0, -3000.0, 400.0, -2000.0, 300.0, 200.0])
    return y, proba, profit


def test_le_protocole_se_charge_et_reste_coherent(cfg):
    assert cfg["seed"] == 42
    assert cfg["promotion"]["metric"] == "roc_auc"
    assert cfg["promotion"]["greater_is_better"] is True
    assert 0 < cfg["promotion"]["threshold"] < 1
    assert {"default", "rapide", "faible"} <= set(cfg["presets"])

    dates = {
        part: {bound: pd.Timestamp(value) for bound, value in cfg["split"][part].items()}
        for part in ("train", "valid", "test")
    }
    assert dates["train"]["start"] < dates["train"]["end"] < dates["valid"]["start"]
    assert dates["valid"]["start"] < dates["valid"]["end"] < dates["test"]["start"]
    # Date que POST /train ne doit jamais atteindre.
    assert dates["test"]["start"] == pd.Timestamp("2015-01-01")


def test_resolve_preset_complete_les_prereglages(cfg):
    assert resolve_preset("rapide", cfg)["sample_size"] == 50_000
    assert resolve_preset("faible", cfg)["features"] == ["loan_amnt", "annual_inc"]

    variante = resolve_preset({"model": "hist_gradient_boosting"}, cfg)
    assert variante["name"] == "custom"
    assert variante["model"] == "hist_gradient_boosting"
    assert variante["variables"] == cfg["reference"]["variables"]
    assert variante["transformation"] == cfg["reference"]["transformation"]


@pytest.mark.parametrize(
    "preset", ["inconnu", {"model": "svm"}, {"variables": "V3"}, {"sample_size": 0}]
)
def test_resolve_preset_refuse_les_valeurs_non_permises(cfg, preset):
    with pytest.raises(ValueError):
        resolve_preset(preset, cfg)


@pytest.mark.parametrize(
    "preset", ["default", "rapide", "faible", {"model": "hist_gradient_boosting"}]
)
def test_run_pipeline_renvoie_un_resultat_complet(cfg, petits_prets, petit_contrat, preset):
    result = run_pipeline(petits_prets, preset, cfg=cfg, contract=petit_contrat)

    assert isinstance(result, PipelineResult)
    assert set(result.metrics) == set(METRIC_KEYS)
    assert result.metric_name == cfg["promotion"]["metric"]
    assert result.metric_value == result.metrics[result.metric_name]
    assert 0 <= result.metric_value <= 1
    assert 0 < result.decision_threshold < 1
    assert result.n_train > 0 and result.n_valid > 0
    assert result.contract_version == "test"
    assert result.features and set(result.features) <= set(petits_prets.columns)
    if preset == "faible":
        assert result.features == ["loan_amnt", "annual_inc"]

    assert list(result.input_example.columns) == result.features
    proba = result.model.predict_proba(result.input_example)[:, 1]
    assert len(proba) == len(result.input_example)
    assert np.all((proba >= 0) & (proba <= 1))


def test_evaluate_renvoie_toutes_les_cles(validation_a_la_main):
    y, proba, profit = validation_a_la_main
    metrics = evaluate(y, proba, 0.5, profit)
    assert set(metrics) == set(METRIC_KEYS)
    assert all(isinstance(value, float) for value in metrics.values())


def test_evaluate_sans_profit_laisse_le_profit_manquant(validation_a_la_main):
    y, proba, _ = validation_a_la_main
    metrics = evaluate(y, proba, 0.5)
    assert set(metrics) == set(METRIC_KEYS)
    assert math.isnan(metrics["realized_profit"])


def test_choose_decision_threshold_reste_dans_la_grille(cfg, validation_a_la_main):
    y, proba, profit = validation_a_la_main
    seuil = choose_decision_threshold(y, proba, profit, cfg=cfg)
    grille = cfg["decision_threshold"]["grid"]
    assert grille["start"] <= seuil <= grille["stop"]


def test_comparison_tags(cfg):
    assert comparison_tags("modele", "hgb", "P2", cfg) == {
        "axe": "modele",
        "variante": "hgb",
        "lot": "P2",
        "protocole": cfg["protocol_version"],
    }


@pytest.mark.parametrize(
    "axe, variante, lot",
    [("modèle", "hgb", "P2"), ("modele", " ", "P2"), ("modele", "hgb", "P6")],
)
def test_comparison_tags_refuse_les_valeurs_non_permises(cfg, axe, variante, lot):
    with pytest.raises(ValueError):
        comparison_tags(axe, variante, lot, cfg)

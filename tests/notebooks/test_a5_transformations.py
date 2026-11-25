"""Tests de notebooks/a5_transformations.py (tâche A5, lot P4).

Le module n'est pas dans le paquet src : on le charge par son chemin.
Aucun test ne lit de vraies données, ni MLflow, ni PostgreSQL.
"""

import importlib.util
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

_PATH = Path(__file__).resolve().parents[2] / "notebooks" / "a5_transformations.py"
_SPEC = importlib.util.spec_from_file_location("a5_transformations", _PATH)
a5 = importlib.util.module_from_spec(_SPEC)
sys.modules["a5_transformations"] = a5
_SPEC.loader.exec_module(a5)

CFG = a5.PROTOCOL_V1


@pytest.fixture
def spec():
    return a5.FeatureSpec(
        numeric=["loan_amnt", "dti"],
        categorical=["home_ownership"],
        amounts=["loan_amnt"],
        source="test",
    )


@pytest.fixture
def train():
    return pd.DataFrame(
        {
            "loan_amnt": [1000.0, 2000.0, np.nan, 4000.0] * 25,
            "dti": [10.0, 20.0, 30.0, 40.0] * 25,
            "home_ownership": (["RENT"] * 50) + (["OWN"] * 49) + ["NONE"],
        }
    )


# --- T1 -------------------------------------------------------------------------------
def test_t1_impute_la_mediane_standardise_et_encode(spec, train):
    ct = a5.make_t1(spec).fit(train)
    out = pd.DataFrame(ct.transform(train), columns=ct.get_feature_names_out())
    assert not out.isna().any().any()
    # Médiane de loan_amnt sur l'entraînement : 2000 ; après standardisation, moyenne nulle.
    assert out["numeriques__loan_amnt"].mean() == pytest.approx(0, abs=1e-9)
    assert out["numeriques__dti"].std(ddof=0) == pytest.approx(1)
    # Une colonne par modalité, NONE comprise : T1 ne regroupe rien.
    assert {"categorielles__home_ownership_NONE", "categorielles__home_ownership_RENT"} <= set(out)
    assert not any(c.startswith("manquants__") for c in out)


def test_t1_modalite_inconnue_donne_des_zeros(spec, train):
    ct = a5.make_t1(spec).fit(train)
    new = train.head(1).assign(home_ownership="MARS")
    out = pd.DataFrame(ct.transform(new), columns=ct.get_feature_names_out())
    assert out.filter(like="home_ownership").to_numpy().sum() == 0


def test_t1_impute_le_mode_des_categorielles(spec, train):
    data = train.copy()
    data.loc[99, "home_ownership"] = np.nan  # la ligne NONE : il reste 50 RENT et 49 OWN
    ct = a5.make_t1(spec).fit(data)
    out = pd.DataFrame(ct.transform(data.tail(1)), columns=ct.get_feature_names_out())
    assert out.iloc[0]["categorielles__home_ownership_RENT"] == 1


# --- T2 -------------------------------------------------------------------------------
def test_t2_ajoute_un_indicateur_seulement_pour_les_colonnes_manquantes_a_l_entrainement(spec, train):
    ct = a5.make_t2(spec).fit(train)
    names = list(ct.get_feature_names_out())
    assert "manquants__missingindicator_loan_amnt" in names
    assert not any("missingindicator_dti" in n for n in names)
    out = pd.DataFrame(ct.transform(train), columns=names)
    assert out["manquants__missingindicator_loan_amnt"].sum() == 25
    # Un manquant nouveau en validation, sur une colonne sans indicateur, ne casse rien.
    valid = train.head(2).assign(dti=np.nan)
    assert ct.transform(valid).shape[1] == len(names)


def test_t2_passe_les_montants_au_log1p(spec, train):
    ct = a5.make_t2(spec).fit(train)
    logged = np.log1p(train["loan_amnt"].fillna(train["loan_amnt"].median()))
    expected = (logged - logged.mean()) / logged.std(ddof=0)
    out = pd.DataFrame(ct.transform(train), columns=ct.get_feature_names_out())
    np.testing.assert_allclose(out["montants__loan_amnt"], expected, atol=1e-9)
    # dti n'est pas un montant : pas de log.
    assert "numeriques__dti" in out


def test_t2_montant_negatif_devient_manquant():
    out = a5._log1p_nonnegative(np.array([[-5.0], [0.0], [np.e - 1]]))
    assert np.isnan(out[0, 0])
    np.testing.assert_allclose(out[1:, 0], [0.0, 1.0])


def test_t2_regroupe_les_modalites_rares_et_inconnues(spec, train):
    # NONE vaut 1/100 = 1 % des lignes. scikit-learn ne déclare rare qu'une modalité
    # strictement sous min_frequency : avec 1 %, NONE garde sa colonne ; avec 2 %, elle
    # rejoint la colonne des modalités rares.
    ct = a5.make_t2(spec).fit(train)
    names = list(ct.get_feature_names_out())
    ct2 = a5.make_t2(spec, min_frequency=0.02).fit(train)
    names2 = list(ct2.get_feature_names_out())
    assert "categorielles__home_ownership_NONE" in names
    assert "categorielles__home_ownership_NONE" not in names2
    assert "categorielles__home_ownership_infrequent_sklearn" in names2
    new = train.head(1).assign(home_ownership="MARS")
    out = pd.DataFrame(ct2.transform(new), columns=names2)
    assert out.loc[0, "categorielles__home_ownership_infrequent_sklearn"] == 1


@pytest.mark.parametrize("name", ["T1", "T2"])
def test_pipeline_complet_donne_des_probabilites(name, spec, train):
    y = np.array([0, 1] * 50)
    pipe = a5.make_pipeline(name, spec, seed=42).fit(train, y)
    proba = pipe.predict_proba(train)[:, 1]
    assert proba.shape == (100,) and ((0 <= proba) & (proba <= 1)).all()
    assert pipe.named_steps["model"].random_state == 42


# --- Variables ------------------------------------------------------------------------
def test_feature_spec_lit_le_format_du_contrat_d_illustration():
    contract = {
        "version": "illustration",
        "features": [
            {"name": "loan_amnt", "type": "numeric", "unit": "dollars"},
            {"name": "annual_inc", "type": "numeric", "unit": "dollars par an"},
            {"name": "dti", "type": "numeric", "unit": "pourcentage"},
            {"name": "home_ownership", "type": "categorical"},
        ],
    }
    spec = a5.feature_spec(contract)
    assert spec.numeric == ["loan_amnt", "annual_inc", "dti"]
    assert spec.amounts == ["loan_amnt", "annual_inc"]
    assert spec.categorical == ["home_ownership"]
    assert spec.source == "contrat illustration"


def test_feature_spec_refuse_une_colonne_interdite():
    with pytest.raises(ValueError, match="interdite"):
        a5.feature_spec({"features": [{"name": "int_rate", "type": "numeric"}]})


def test_v1_provisoire_ne_contient_aucune_colonne_interdite():
    spec = a5.feature_spec(None)
    assert not set(spec.all) & set(a5.FORBIDDEN_COLUMNS)
    assert spec.source == "V1 provisoire (A5)"


# --- Protocole ------------------------------------------------------------------------
def test_decoupage_temporel_bornes_incluses():
    df = pd.DataFrame({"issue_d": pd.to_datetime(["2007-06-01", "2013-12-31", "2014-01-01", "2014-12-31"])})
    train, valid = a5.temporal_split_local(df, CFG)
    assert len(train) == 2 and len(valid) == 2
    assert train["issue_d"].max() < pd.Timestamp("2014-01-01")


def test_decoupage_refuse_2015():
    df = pd.DataFrame({"issue_d": pd.to_datetime(["2014-06-01", "2015-01-01"])})
    with pytest.raises(ValueError, match="test"):
        a5.temporal_split_local(df, CFG)


def test_clean_local_construit_la_cible_et_ecarte_les_prets_en_cours():
    df = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Charged Off", "Current", "Default",
                            "Does not meet the credit policy. Status:Fully Paid"],
            "issue_d": ["2010-01-01"] * 5,
            "term": [" 36 months", " 36 months", " 36 months", " 36 months", " 60 months"],
        }
    )
    out = a5.clean_local(df, CFG)
    assert out["target"].tolist() == [0, 1, 1]
    assert "Current" not in set(out["loan_status"])


def test_evaluate_local_refuse_a_partir_du_seuil():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.5, 0.5, 0.9])
    profit = np.array([100.0, 50.0, -300.0, -500.0])
    m = a5.evaluate_local(y, proba, 0.5, profit)
    assert set(m) == set(a5.METRIC_KEYS)
    assert m["acceptance_rate"] == 0.25  # seul 0,1 < 0,5 est accepté
    assert m["realized_profit"] == 100.0
    assert m["recall"] == 1.0 and m["precision"] == pytest.approx(2 / 3)
    assert np.isnan(a5.evaluate_local(y, proba, 0.5)["realized_profit"])


def test_seuil_maximise_le_profit_et_garde_le_plus_petit():
    y = np.array([0, 1, 0])
    proba = np.array([0.2, 0.6, 0.3])
    profit = np.array([100.0, -500.0, 50.0])
    # Accepter 0,2 et 0,3 sans 0,6 : tout seuil de 0,31 à 0,60. Le plus petit est 0,31.
    assert a5.choose_threshold_local(y, proba, profit, CFG) == pytest.approx(0.31)
    grid = a5.threshold_grid(CFG)
    assert grid[0] == 0.01 and grid[-1] == 0.99 and len(grid) == 99


def test_bootstrap_apparie_ecart_nul_pour_deux_variantes_identiques():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    res = a5.paired_bootstrap_auc(y, p, p, n_boot=50)
    assert res["diff"] == 0 and res["low"] == 0 and res["high"] == 0


def test_is_stub_reconnait_l_avertissement_des_corps_factices():
    import logging

    def factice():
        logging.getLogger("src.training.x").warning("evaluate : corps factice (C2), remplacé par M2.")

    def reelle():
        return 1

    def casse():
        raise RuntimeError

    assert a5.is_stub(factice) and not a5.is_stub(reelle) and a5.is_stub(casse)
    assert a5.probe_status(factice) == "factice"
    assert a5.probe_status(reelle) == "reelle"
    assert a5.probe_status(casse).startswith("erreur : RuntimeError")


# --- Prêts factices -------------------------------------------------------------------
def test_prets_factices_sans_2015_et_conformes_au_format_brut():
    df = a5.make_synthetic_loans_a5(2000, seed=1)
    assert pd.to_datetime(df["issue_d"]).max() < pd.Timestamp("2015-01-01")
    spec = a5.feature_spec(None)
    assert set(spec.all) <= set(df.columns)
    assert {"loan_status", "issue_d", "funded_amnt", "total_pymnt"} <= set(df.columns)
    assert df["mort_acc"][pd.to_datetime(df["issue_d"]).dt.year < 2010].isna().all()
    pd.testing.assert_frame_equal(df, a5.make_synthetic_loans_a5(2000, seed=1))


# --- Compléments issus du test par mutations (vérification 1) -----------------------
def test_t1_impute_la_mediane_et_ne_passe_pas_au_log(spec, train):
    # Médiane de 1000, 2000, 4000 (répétés) : 2000 ; la moyenne vaudrait 2333.
    ct = a5.make_t1(spec).fit(train)
    filled = train["loan_amnt"].fillna(2000.0)
    expected = (filled - filled.mean()) / filled.std(ddof=0)
    out = pd.DataFrame(ct.transform(train), columns=ct.get_feature_names_out())
    np.testing.assert_allclose(out["numeriques__loan_amnt"], expected, atol=1e-9)


@pytest.mark.parametrize("make", [a5.make_t1, a5.make_t2])
def test_les_colonnes_hors_variables_sont_ecartees(make, spec, train):
    data = train.assign(target=[0, 1] * 50, funded_amnt=1000.0)
    names = make(spec).fit(data).get_feature_names_out()
    assert not any("target" in n or "funded_amnt" in n for n in names)


def test_t2_indicateur_pour_une_categorielle_manquante(spec, train):
    data = train.copy()
    data.loc[0, "home_ownership"] = np.nan
    names = a5.make_t2(spec).fit(data).get_feature_names_out()
    assert "manquants__missingindicator_home_ownership" in names


def test_t2_transforme_chaque_ligne_avec_les_statistiques_de_l_entrainement(spec, train):
    # Une ligne transformée seule doit donner le même résultat que dans un lot.
    ct = a5.make_t2(spec, min_frequency=0.02).fit(train)
    full = ct.transform(train)
    one_by_one = np.vstack([ct.transform(train.iloc[[i]]) for i in range(4)])
    np.testing.assert_allclose(one_by_one, full[:4])


def test_make_pipeline_associe_chaque_nom_a_sa_transformation(spec, train):
    y = np.array([0, 1] * 50)
    names = {
        n: set(a5.make_pipeline(n, spec, seed=42).fit(train, y).named_steps["transformation"].get_feature_names_out())
        for n in ("T1", "T2")
    }
    assert "numeriques__loan_amnt" in names["T1"]
    assert not any(n.startswith(("montants__", "manquants__")) for n in names["T1"])
    assert {"montants__loan_amnt", "manquants__missingindicator_loan_amnt"} <= names["T2"]


@pytest.mark.parametrize("name", ["grade", "sub_grade", "int_rate", "installment", "funded_amnt_inv"])
def test_feature_spec_refuse_chaque_colonne_interdite(name):
    with pytest.raises(ValueError, match="interdite"):
        a5.feature_spec({"features": [{"name": name, "type": "numeric"}]})


def test_montant_par_l_unite_sans_casse_ou_par_le_nom_exact():
    spec = a5.feature_spec(
        {
            "version": "x",
            "features": [
                {"name": "loan_amnt", "type": "numeric", "unit": "comptes"},  # nom listé
                {"name": "revenu", "type": "numeric", "unit": " Dollars par an "},  # unité, casse
                {"name": "solde", "type": "numeric", "unit": "USD"},
                {"name": "inc_ratio", "type": "numeric", "unit": "pourcentage"},  # ni l'un ni l'autre
                {"name": "funded_amnt_x", "type": "numeric", "unit": None},
            ],
        }
    )
    assert spec.amounts == ["loan_amnt", "revenu", "solde"]
    assert a5.feature_spec(None).amounts == ["loan_amnt", "annual_inc", "revol_bal"]


def test_to_sklearn_types_remplace_les_types_nullables():
    X = pd.DataFrame(
        {
            "n": pd.array([1, None, 3], dtype="Int64"),
            "f": pd.array([1.5, None, 2.0], dtype="Float64"),
            "b": [True, False, True],
            "s": pd.array(["RENT", None, "OWN"], dtype="string"),
        }
    )
    out = a5.to_sklearn_types(X)
    assert all(out[c].dtype == "float64" for c in ("n", "f", "b"))
    assert out["s"].dtype == object
    assert np.isnan(out.loc[1, "n"]) and np.isnan(out.loc[1, "f"])
    assert out.loc[1, "s"] is np.nan
    assert out["b"].tolist() == [1.0, 0.0, 1.0]


@pytest.mark.parametrize("name", ["T1", "T2"])
def test_pipeline_accepte_les_types_nullables_de_pandas(name, spec, train):
    data = train.astype({"loan_amnt": "Float64", "home_ownership": "string"})
    data["dti"] = data["dti"].astype("Int64")
    data.loc[1, "home_ownership"] = pd.NA
    y = np.array([0, 1] * 50)
    proba = a5.make_pipeline(name, spec, seed=42).fit(data, y).predict_proba(data)[:, 1]
    assert np.isfinite(proba).all()


def test_decoupage_un_pret_du_31_decembre_a_midi_reste_dans_l_entrainement():
    df = pd.DataFrame({"issue_d": pd.to_datetime(["2013-12-31 12:00", "2014-12-31 23:00"])})
    train, valid = a5.temporal_split_local(df, CFG)
    assert len(train) == 1 and len(valid) == 1


def test_probe_status_entend_l_avertissement_meme_logs_coupes_et_restaure_l_etat():
    import logging

    def factice():
        logging.getLogger("src.training.x").warning("clean : corps factice (C1).")

    src_logger = logging.getLogger("src")
    old_level = src_logger.level
    try:
        src_logger.setLevel(logging.ERROR)
        logging.disable(logging.CRITICAL)
        assert a5.probe_status(factice) == "factice"
        assert logging.root.manager.disable == logging.CRITICAL
        assert src_logger.level == logging.ERROR
    finally:
        logging.disable(logging.NOTSET)
        src_logger.setLevel(old_level)


def test_clean_local_perimetre_doublons_et_entree_intacte():
    df = pd.DataFrame(
        {
            "loan_status": ["Fully Paid", "Fully Paid", "Charged Off", "Charged Off", "Fully Paid"],
            "issue_d": ["2007-05-31", "2007-06-01", "2015-12-31", "2015-12-31", "2016-01-01"],
            "term": [" 36 months"] * 5,
        }
    )
    before = df.copy()
    out = a5.clean_local(df, CFG)
    # Bornes du périmètre incluses, doublon exact retiré.
    assert out["issue_d"].dt.strftime("%Y-%m-%d").tolist() == ["2007-06-01", "2015-12-31"]
    pd.testing.assert_frame_equal(df, before)


def test_decoupage_range_chaque_pret_dans_sa_periode():
    df = pd.DataFrame(
        {
            "issue_d": pd.to_datetime(["2007-05-31", "2007-06-01", "2013-12-31", "2014-01-01", "2014-12-31"]),
            "k": range(5),
        }
    )
    train, valid = a5.temporal_split_local(df, CFG)
    assert train["k"].tolist() == [1, 2]
    assert valid["k"].tolist() == [3, 4]


def test_evaluate_local_scores_calcules_sur_les_probabilites():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])
    m = a5.evaluate_local(y, proba, 0.5)
    assert m["roc_auc"] == pytest.approx(0.75)
    assert m["pr_auc"] == pytest.approx(0.5 + 0.5 * 2 / 3)
    assert m["brier"] == pytest.approx((0.01 + 0.36 + 0.36 + 0.01) / 4)


def test_seuil_refuse_des_longueurs_differentes():
    with pytest.raises(ValueError, match="longueur"):
        a5.choose_threshold_local([0, 1], [0.2, 0.6, 0.3], [1.0, 2.0, 3.0], CFG)


def test_bootstrap_signe_et_quantiles_2_5_et_97_5():
    # Refait les tirages avec la même graine : à adapter si l'ordre des tirages change.
    rng = np.random.default_rng(0)
    n = 2000
    y = rng.integers(0, 2, n)
    bon = y + rng.normal(0, 0.8, n)
    bruit = rng.random(n)
    res = a5.paired_bootstrap_auc(y, bruit, bon, n_boot=200, seed=1)
    assert res["diff"] == pytest.approx(roc_auc_score(y, bon) - roc_auc_score(y, bruit))
    assert res["diff"] > 0.1  # b meilleur que a : écart positif (b - a)
    # Référence : mêmes indices pour a et b, quantiles 2,5 % et 97,5 %.
    ref = np.random.default_rng(1)
    diffs = []
    for _ in range(200):
        idx = ref.integers(0, n, n)
        diffs.append(roc_auc_score(y[idx], bon[idx]) - roc_auc_score(y[idx], bruit[idx]))
    assert res["low"] == pytest.approx(np.quantile(diffs, 0.025))
    assert res["high"] == pytest.approx(np.quantile(diffs, 0.975))
    assert res["n_boot"] == 200


def test_bootstrap_ecarte_les_tirages_a_une_seule_classe():
    y = np.array([0, 1, 0, 1])
    a = np.array([0.1, 0.9, 0.2, 0.8])
    b = np.array([0.2, 0.7, 0.4, 0.6])
    res = a5.paired_bootstrap_auc(y, a, b, n_boot=200, seed=0)
    assert 0 < res["n_boot"] < 200


def test_prets_factices_ont_les_manquants_et_modalites_rares_annonces():
    df = a5.make_synthetic_loans_a5(20_000, seed=3)
    assert pd.to_datetime(df["issue_d"]).min() >= pd.Timestamp("2007-06-01")
    assert df["emp_length"].isna().mean() > 0.02
    assert df["revol_util"].isna().mean() > 0.005
    assert (df["addr_state"].value_counts(normalize=True) < a5.RARE_MIN_FREQUENCY).any()
    assert set(a5.clean_local(df, CFG)["target"]) == {0, 1}


# --- Compléments de la vérification 2 ------------------------------------------------
def test_t2_sans_ajout_redonne_t1(spec, train):
    t1 = a5.make_t1(spec).fit(train)
    nu = a5.make_t2(spec, log_amounts=False, indicators=False, rare=False).fit(train)
    np.testing.assert_allclose(nu.transform(train), t1.transform(train))
    assert list(nu.get_feature_names_out()) == list(t1.get_feature_names_out())


def test_t2_options_retirent_un_ajout_a_la_fois(spec, train):
    seul_log = set(a5.make_t2(spec, indicators=False, rare=False).fit(train).get_feature_names_out())
    assert "montants__loan_amnt" in seul_log and not any(n.startswith("manquants__") for n in seul_log)
    sans_ind = set(a5.make_t2(spec, indicator_exclude=["loan_amnt"]).fit(train).get_feature_names_out())
    assert not any("missingindicator_loan_amnt" in n for n in sans_ind)


def test_make_pipeline_from_utilise_la_transformation_donnee(spec, train):
    y = np.array([0, 1] * 50)
    ct = a5.make_t1(spec)
    pipe = a5.make_pipeline_from(ct, seed=42).fit(train, y)
    assert pipe.named_steps["transformation"] is ct
    np.testing.assert_allclose(
        pipe.predict_proba(train), a5.make_pipeline("T1", spec, seed=42).fit(train, y).predict_proba(train)
    )


def test_bootstrap_generique_sur_le_brier():
    from sklearn.metrics import brier_score_loss

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000)
    bon = np.clip(y * 0.7 + 0.15 + rng.normal(0, 0.05, 1000), 0, 1)
    mauvais = np.full(1000, 0.5)
    res = a5.paired_bootstrap(y, mauvais, bon, brier_score_loss, n_boot=100, seed=0)
    assert res["diff"] < 0 and res["high"] < 0  # b nettement meilleur : Brier plus bas


def test_clean_local_met_les_valeurs_impossibles_a_manquant():
    df = pd.DataFrame(
        {
            "loan_status": ["Fully Paid"] * 3,
            "issue_d": ["2010-01-01", "2010-02-01", "2010-03-01"],
            "annual_inc": [0.0, -5.0, 40000.0],
            "dti": [-1.0, 10.0, 20.0],
        }
    )
    out = a5.clean_local(df, CFG)
    assert out["annual_inc"].isna().tolist() == [True, True, False]
    assert out["dti"].isna().tolist() == [True, False, False]


def test_origine_glissante_bornes_et_refus_du_test():
    df = pd.DataFrame(
        {"issue_d": pd.to_datetime(["2007-06-01 00:00", "2011-12-31 10:00", "2012-01-01 00:00", "2012-12-31 00:00", "2013-05-01 00:00"])}
    )
    [(year, tr, va)] = a5.rolling_origin(df, CFG, [2012])
    assert year == 2012 and len(tr) == 2 and len(va) == 2
    with pytest.raises(ValueError):
        a5.rolling_origin(df, CFG, [2015])
    with pytest.raises(ValueError):
        a5.rolling_origin(pd.concat([df, pd.DataFrame({"issue_d": [pd.Timestamp("2015-02-01")]})]), CFG, [2012])


# --- Compléments issus de la vérification 2 (exécution) -------------------------------
def test_feature_spec_lit_un_contrat_en_dictionnaire_et_le_type_integer():
    spec = a5.feature_spec(
        {
            "version": "d",
            "features": {
                "loan_amnt": {"type": "numeric", "unit": "USD"},
                "open_acc": {"type": "integer", "unit": "comptes"},
                "purpose": {"type": "categorical"},
            },
        }
    )
    assert spec.numeric == ["loan_amnt", "open_acc"]
    assert spec.categorical == ["purpose"]
    assert spec.amounts == ["loan_amnt"]
    assert spec.source == "contrat d"


@pytest.mark.parametrize("kind", [None, "date", "datetime"])
def test_feature_spec_refuse_un_type_inconnu(kind):
    with pytest.raises(ValueError, match="Type inconnu"):
        a5.feature_spec({"features": [{"name": "x", "type": kind}]})


def test_t2_montant_negatif_en_validation_recoit_la_mediane_de_l_entrainement(spec, train):
    ct = a5.make_t2(spec).fit(train)
    names = list(ct.get_feature_names_out())
    median = train["loan_amnt"].median()
    neg = pd.DataFrame(ct.transform(train.head(1).assign(loan_amnt=-50.0)), columns=names)
    med = pd.DataFrame(ct.transform(train.head(1).assign(loan_amnt=median)), columns=names)
    assert neg.loc[0, "montants__loan_amnt"] == pytest.approx(med.loc[0, "montants__loan_amnt"])
    # Le montant négatif devient manquant après le log : l'indicateur de loan_amnt ne le voit pas,
    # car MissingIndicator lit la colonne brute. clean() met déjà ces valeurs à manquant avant.
    assert neg.loc[0, "manquants__missingindicator_loan_amnt"] == 0


def test_probe_status_restaure_l_etat_des_logs_meme_si_l_appel_echoue():
    src_logger = logging.getLogger("src")
    old_level, old_handlers = src_logger.level, list(src_logger.handlers)
    try:
        src_logger.setLevel(logging.ERROR)
        logging.disable(logging.CRITICAL)

        def casse():
            logging.getLogger("src.training.x").warning("corps factice")
            raise KeyError("earliest_cr_line")

        assert a5.probe_status(casse) == "erreur : KeyError : 'earliest_cr_line'"
        assert logging.root.manager.disable == logging.CRITICAL
        assert src_logger.level == logging.ERROR
        assert src_logger.handlers == old_handlers
    finally:
        logging.disable(logging.NOTSET)
        src_logger.setLevel(old_level)


def test_protocol_v1_identique_a_config_training_quand_il_existe():
    path = Path(__file__).resolve().parents[2] / "config" / "training.yaml"
    if not path.exists():
        pytest.skip("config/training.yaml absent (PR #44 non fusionnée)")
    import yaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in (*a5.PROTOCOL_KEYS_USED, "scope", "comparison_tags"):
        assert cfg[key] == a5.PROTOCOL_V1[key], key


@pytest.mark.parametrize("kind", ["int", "float64", "boolean"])
def test_feature_spec_accepte_les_types_numeriques_courants(kind):
    assert a5.feature_spec({"features": [{"name": "x", "type": kind}]}).numeric == ["x"]


def test_seuil_garde_le_plus_petit_quand_tous_les_profits_sont_negatifs():
    # Accepter un prêt fait perdre de l'argent : le meilleur total (0) est atteint en
    # n'acceptant rien, dès le premier seuil de la grille.
    y = np.array([1, 1])
    proba = np.array([0.2, 0.6])
    profit = np.array([-10.0, -20.0])
    assert a5.choose_threshold_local(y, proba, profit, CFG) == pytest.approx(0.01)


def test_evaluate_local_sans_aucun_refus():
    y = np.array([0, 1, 0, 1])
    proba = np.array([0.1, 0.8, 0.3, 0.6])
    m = a5.evaluate_local(y, proba, 0.99, np.array([1.0, -2.0, 3.0, -4.0]))
    assert m["acceptance_rate"] == 1.0
    assert m["precision"] == 0.0 and m["recall"] == 0.0
    assert m["realized_profit"] == pytest.approx(-2.0)


@pytest.mark.parametrize("name", ["T1", "T2"])
def test_make_pipeline_donne_les_memes_probabilites_a_chaque_ajustement(name, spec, train):
    y = np.array([0, 1] * 50)
    p1 = a5.make_pipeline(name, spec, seed=42).fit(train, y).predict_proba(train)
    p2 = a5.make_pipeline(name, spec, seed=42).fit(train, y).predict_proba(train)
    np.testing.assert_array_equal(p1, p2)


@pytest.mark.skipif(os.environ.get("A5_NOTEBOOK") != "1", reason="exécution complète du notebook : A5_NOTEBOOK=1")
def test_notebook_05_s_execute_de_zero_sans_erreur(tmp_path, monkeypatch):
    # Lancer avec : A5_NOTEBOOK=1 uv run pytest tests/notebooks -q (uv place .venv/Scripts dans PATH,
    # sinon le noyau python3 peut être celui du système, sans mlflow).
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    nb_dir = Path(__file__).resolve().parents[2] / "notebooks"
    monkeypatch.setenv("TEMP", str(tmp_path))
    monkeypatch.setenv("TMP", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    nb = nbformat.read(nb_dir / "05_comparaison_transformations.ipynb", as_version=4)
    nbclient.NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(nb_dir)}}).execute()
    texts = []
    for cell in nb.cells:
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", out.get("evalue")
            texts.append(out.get("text", ""))
    bilan = "".join(texts)
    assert "Règle de choix proposée" in bilan
    assert "échec de l'entraînement" not in bilan

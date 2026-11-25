"""Tâche A3 : POST /train protégé par la clé X-API-Key.

train_and_promote (tâche R2, lot P3) est remplacée par de fausses fonctions, injectées par
app.dependency_overrides ou par un faux module src.registry.promotion. Aucun test ne demande
MLflow ni PostgreSQL.
"""

import json
import logging
import subprocess
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import main, schemas, security, train

CONTRACT = {
    "version": "test-a3",
    "features": [
        {"name": "loan_amnt", "type": "numeric", "min": 1000, "max": 40000},
        {"name": "home_ownership", "type": "categorical", "allowed": ["RENT", "OWN"]},
    ],
}
EXAMPLES = {"prudent": {"loan_amnt": 5000, "home_ownership": "OWN"}}
KEY = "cle-entrainement-Zq8-test"
PROMOTED, REJECTED = schemas.TrainResponse.model_config["json_schema_extra"]["examples"]
WAIT_SECONDS = 10


@pytest.fixture(autouse=True)
def configured_key(monkeypatch):
    monkeypatch.setenv(security.API_KEY_ENV, KEY)


@pytest.fixture
def app(tmp_path):
    contract_path = tmp_path / "features.yaml"
    examples_path = tmp_path / "examples.json"
    contract_path.write_text(json.dumps(CONTRACT), encoding="utf-8")  # le JSON est du YAML valide
    examples_path.write_text(json.dumps(EXAMPLES), encoding="utf-8")
    return main.create_app(contract_path, examples_path, database_check=lambda: True)


class FakeTrainer:
    """Fausse train_and_promote : note les préréglages reçus et renvoie le résultat prévu."""

    def __init__(self, result=PROMOTED, error: BaseException | None = None):
        self.result = result
        self.error = error
        self.presets: list[str] = []
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(self, preset):
        self.presets.append(preset)
        self.started.set()
        assert self.release.wait(WAIT_SECONDS), "le test n'a pas libéré l'entraînement"
        if self.error is not None:
            raise self.error
        return self.result


def use(app, trainer):
    app.dependency_overrides[train.get_train_and_promote] = lambda: trainer
    return trainer


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def post(client, key=KEY, **kwargs):
    headers = {} if key is None else {"X-API-Key": key}
    return client.post("/train", headers=headers, **kwargs)


def assert_error(response, status, code):
    assert response.status_code == status
    error = schemas.ErrorResponse.model_validate(response.json())
    assert error.code == code
    return error


# --- clé ---------------------------------------------------------------------


def test_without_header_answers_401_in_the_common_format(app, client):
    trainer = use(app, FakeTrainer())
    response = post(client, key=None, json={"preset": "rapide"})
    error = assert_error(response, 401, schemas.ErrorCode.UNAUTHORIZED)
    assert "X-API-Key" in error.detail
    assert response.headers["WWW-Authenticate"] == "APIKey"
    assert trainer.presets == []


def test_an_empty_header_counts_as_absent(app, client):
    use(app, FakeTrainer())
    assert_error(post(client, key=""), 401, schemas.ErrorCode.UNAUTHORIZED)


@pytest.mark.parametrize("wrong", ["mauvaise", KEY[:-1], KEY + "x", KEY.upper(), " " * 3 + "a"])
def test_a_wrong_key_answers_403(app, client, wrong):
    trainer = use(app, FakeTrainer())
    assert_error(post(client, key=wrong), 403, schemas.ErrorCode.FORBIDDEN)
    assert trainer.presets == []


@pytest.mark.parametrize("server_value", [None, "", "   "])
def test_an_unconfigured_server_keeps_the_route_closed(app, client, monkeypatch, server_value, caplog):
    if server_value is None:
        monkeypatch.delenv(security.API_KEY_ENV)
    else:
        monkeypatch.setenv(security.API_KEY_ENV, server_value)
    trainer = use(app, FakeTrainer())
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        unconfigured = post(client, key="n-importe-quoi")
        blank = post(client, key=server_value or "x")
    assert_error(unconfigured, 403, schemas.ErrorCode.FORBIDDEN)
    assert_error(blank, 403, schemas.ErrorCode.FORBIDDEN)
    assert trainer.presets == []
    assert "TRAIN_API_KEY n'est pas configurée" in caplog.text


def test_unconfigured_and_wrong_key_give_the_same_answer(app, client, monkeypatch):
    use(app, FakeTrainer())
    wrong = post(client, key="mauvaise").json()
    monkeypatch.delenv(security.API_KEY_ENV)
    unconfigured = post(client, key="mauvaise").json()
    assert wrong == unconfigured


def test_the_right_key_trains_with_the_requested_preset(app, client):
    trainer = use(app, FakeTrainer())
    response = post(client, json={"preset": "rapide"})
    assert response.status_code == 200
    assert schemas.TrainResponse.model_validate(response.json()) == schemas.TrainResponse.model_validate(PROMOTED)
    assert trainer.presets == ["rapide"]


@pytest.mark.parametrize("preset", ["default", "rapide", "faible"])
def test_each_preset_reaches_train_and_promote(app, client, preset):
    trainer = use(app, FakeTrainer(result=REJECTED))
    assert post(client, json={"preset": preset}).status_code == 200
    assert trainer.presets == [preset]


def test_spaces_around_the_configured_key_are_ignored(app, client, monkeypatch):
    monkeypatch.setenv(security.API_KEY_ENV, f"  {KEY}\r\n")
    use(app, FakeTrainer())
    assert post(client).status_code == 200


def test_a_non_ascii_key_sent_in_utf8_is_recognised(app, client, monkeypatch):
    monkeypatch.setenv(security.API_KEY_ENV, "clé-été")
    use(app, FakeTrainer())
    response = client.post("/train", headers={"X-API-Key": "clé-été".encode()})
    assert response.status_code == 200
    assert client.post("/train", headers={"X-API-Key": "cle-ete"}).status_code == 403


def test_the_key_is_checked_before_the_body(app, client):
    use(app, FakeTrainer())
    assert_error(post(client, key=None, json={"preset": "lent"}), 401, schemas.ErrorCode.UNAUTHORIZED)
    assert_error(post(client, key="mauvaise", json={"preset": "lent"}), 403, schemas.ErrorCode.FORBIDDEN)
    error = assert_error(post(client, json={"preset": "lent"}), 422, schemas.ErrorCode.VALIDATION_ERROR)
    assert error.errors[0].field == "preset"


def test_keys_never_appear_in_responses_or_logs(app, client, monkeypatch, caplog):
    wrong = "mauvaise-cle-9F3"
    use(app, FakeTrainer())
    # logger="neuraxion" : create_app règle ce logger sur INFO, le niveau DEBUG doit y être forcé.
    with caplog.at_level(logging.DEBUG, logger="neuraxion"):
        responses = [post(client, key=wrong), post(client), post(client, key=None)]
        monkeypatch.delenv(security.API_KEY_ENV)
        responses.append(post(client, key=KEY))
    assert [r.status_code for r in responses] == [403, 200, 401, 403]
    for response in responses:
        assert KEY not in response.text and wrong not in response.text
        assert KEY not in str(response.headers) and wrong not in str(response.headers)
    assert KEY not in caplog.text and wrong not in caplog.text
    assert "POST /train refusé (403) : clé X-API-Key fausse." in caplog.text


def test_keys_match_compares_digests_in_constant_time(monkeypatch):
    calls = []
    original = security.hmac.compare_digest
    monkeypatch.setattr(security.hmac, "compare_digest", lambda a, b: calls.append((len(a), len(b))) or original(a, b))
    assert security.keys_match("abc", "abc")
    assert not security.keys_match("ab", "abc")
    assert not security.keys_match("x" * 500, "abc")
    # Les empreintes SHA-256 ont toujours 32 octets : la longueur de la clé n'influe pas.
    assert calls == [(32, 32)] * 3


def test_the_unconfigured_case_still_runs_a_comparison(monkeypatch):
    monkeypatch.delenv(security.API_KEY_ENV)
    calls = []
    monkeypatch.setattr(security, "keys_match", lambda a, b: calls.append(1) or False)
    request = types.SimpleNamespace(headers=types.SimpleNamespace(getlist=lambda name: ["quelconque"]))
    with pytest.raises(security.ApiError):
        security.require_api_key(request, "quelconque")
    assert calls == [1]


@pytest.mark.parametrize(
    "headers",
    [
        [("X-API-Key", KEY), ("X-API-Key", KEY)],
        [("X-API-Key", KEY), ("X-API-Key", "mauvaise")],
        [("X-API-Key", ""), ("X-API-Key", KEY)],
        [("x-api-key", KEY), ("X-API-KEY", KEY)],
    ],
    ids=["identiques", "bonne-puis-mauvaise", "vide-puis-bonne", "casses"],
)
def test_several_key_headers_are_refused(app, client, headers, caplog):
    trainer = use(app, FakeTrainer())
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        response = client.post("/train", headers=headers)
    assert_error(response, 403, schemas.ErrorCode.FORBIDDEN)
    assert trainer.presets == []
    assert "plusieurs en-têtes X-API-Key" in caplog.text


def test_openapi_declares_the_key_and_the_statuses(client):
    document = client.get("/openapi.json").json()
    scheme = document["components"]["securitySchemes"]["TrainApiKey"]
    assert scheme == {**scheme, "type": "apiKey", "in": "header", "name": "X-API-Key"}
    operation = document["paths"]["/train"]["post"]
    assert operation["security"] == [{"TrainApiKey": []}]
    assert {"200", "401", "403", "409", "422", "500", "501"} <= set(operation["responses"])
    assert "requestBody" in operation


def test_get_train_is_not_allowed(client):
    assert_error(client.get("/train"), 405, schemas.ErrorCode.METHOD_NOT_ALLOWED)


# --- adaptateur vers R2 --------------------------------------------------------


def test_without_the_r2_module_the_route_answers_501(client, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, train.PROMOTION_MODULE, None)  # import impossible
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        error = assert_error(post(client), 501, schemas.ErrorCode.NOT_IMPLEMENTED)
    assert "R2" in error.detail and "train_and_promote" in error.detail
    assert "n'existe pas encore" in caplog.text


def test_without_the_r2_module_an_anonymous_client_still_gets_401(client, monkeypatch):
    monkeypatch.setitem(sys.modules, train.PROMOTION_MODULE, None)
    assert_error(post(client, key=None), 401, schemas.ErrorCode.UNAUTHORIZED)
    assert_error(post(client, key="mauvaise"), 403, schemas.ErrorCode.FORBIDDEN)


def test_a_module_without_the_function_answers_501(client, monkeypatch):
    monkeypatch.setitem(sys.modules, train.PROMOTION_MODULE, types.ModuleType(train.PROMOTION_MODULE))
    assert_error(post(client), 501, schemas.ErrorCode.NOT_IMPLEMENTED)


def test_the_lazy_import_calls_the_delivered_function(client, monkeypatch):
    module = types.ModuleType(train.PROMOTION_MODULE)
    module.train_and_promote = FakeTrainer(result=REJECTED)
    monkeypatch.setitem(sys.modules, train.PROMOTION_MODULE, module)
    response = post(client, json={"preset": "faible"})
    assert response.status_code == 200
    assert response.json()["status"] == "rejected"
    assert module.train_and_promote.presets == ["faible"]


def test_sys_exit_while_importing_r2_stays_in_the_common_format(client, monkeypatch, caplog):
    def exiting_import(name):
        raise SystemExit("MESSAGE-SECRET-IMPORT")

    monkeypatch.setattr(train.importlib, "import_module", exiting_import)
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert_error(post(client), 500, schemas.ErrorCode.INTERNAL_ERROR)
    assert "a levé SystemExit" in caplog.text and "MESSAGE-SECRET-IMPORT" not in caplog.text
    assert_error(post(client, key=None), 401, schemas.ErrorCode.UNAUTHORIZED)


def test_a_missing_dependency_inside_r2_is_not_hidden_as_501(client, monkeypatch, caplog):
    def broken_import(name):
        raise ModuleNotFoundError("No module named 'mlflow_absent'", name="mlflow_absent")

    monkeypatch.setattr(train.importlib, "import_module", broken_import)
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert_error(post(client), 500, schemas.ErrorCode.INTERNAL_ERROR)
    assert "ModuleNotFoundError" in caplog.text
    assert "le module mlflow_absent est introuvable" in caplog.text


class ForbidR2Import:
    """Chercheur de modules qui échoue si quelqu'un importe le module de R2."""

    def __init__(self):
        self.attempts = 0

    def find_spec(self, name, path=None, target=None):
        if name == train.PROMOTION_MODULE:
            self.attempts += 1
            raise AssertionError("import de R2 au démarrage")


def test_the_api_starts_without_importing_r2(tmp_path, monkeypatch):
    # create_app ne doit pas importer le module de R2 (ni MLflow par lui) : seul l'appel de la route le fait.
    monkeypatch.delitem(sys.modules, train.PROMOTION_MODULE, raising=False)
    finder = ForbidR2Import()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    application = main.create_app(*_paths(tmp_path / "demarrage"), database_check=lambda: True)
    assert finder.attempts == 0
    with TestClient(application) as test_client:
        assert test_client.get("/health").status_code == 200
        assert finder.attempts == 0
        # L'appel de la route, lui, importe le module : le chercheur lève, ce qui donne une 500.
        assert post(test_client).status_code == 500
    assert finder.attempts == 1


# --- un seul entraînement à la fois ---------------------------------------------


def test_a_second_training_during_the_first_answers_409(app, client):
    trainer = use(app, FakeTrainer())
    trainer.release.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(post, client, json={"preset": "rapide"})
        assert trainer.started.wait(WAIT_SECONDS)
        error = assert_error(post(client), 409, schemas.ErrorCode.TRAINING_IN_PROGRESS)
        assert "déjà en cours" in error.detail
        trainer.release.set()
        assert first.result(timeout=WAIT_SECONDS).status_code == 200
    assert trainer.presets == ["rapide"]
    assert post(client).status_code == 200  # le verrou est libéré


def test_simultaneous_requests_start_a_single_training(app, client):
    trainer = use(app, FakeTrainer())
    trainer.release.clear()
    count = 12
    barrier = threading.Barrier(count)

    def send():
        barrier.wait(WAIT_SECONDS)
        return post(client).status_code

    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(send) for _ in range(count)]
        assert trainer.started.wait(WAIT_SECONDS)
        # Les autres requêtes ont répondu 409 sans attendre la fin de l'entraînement.
        deadline = time.monotonic() + WAIT_SECONDS
        while sum(f.done() for f in futures) < count - 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        trainer.release.set()
        statuses = sorted(f.result(timeout=WAIT_SECONDS) for f in futures)
    assert statuses == [200] + [409] * (count - 1)
    assert len(trainer.presets) == 1


def test_the_event_loop_keeps_serving_during_a_training(app, client):
    trainer = use(app, FakeTrainer())
    trainer.release.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(post, client)
        assert trainer.started.wait(WAIT_SECONDS)
        started = time.perf_counter()
        assert client.get("/health").status_code == 200
        assert client.get("/schema").status_code == 200
        assert time.perf_counter() - started < 2
        trainer.release.set()
        assert first.result(timeout=WAIT_SECONDS).status_code == 200


def test_the_lock_is_released_after_a_failed_training(app, client, caplog):
    use(app, FakeTrainer(error=RuntimeError("données absentes : /chemin/secret")))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert_error(post(client), 500, schemas.ErrorCode.INTERNAL_ERROR)
    assert "RuntimeError" in caplog.text and "/chemin/secret" not in caplog.text
    use(app, FakeTrainer())
    assert post(client).status_code == 200


def test_each_application_has_its_own_lock(app, tmp_path):
    other = main.create_app(*_paths(tmp_path / "autre"), database_check=lambda: True)
    assert train.training_lock(app.state) is train.training_lock(app.state)
    assert train.training_lock(app.state) is not train.training_lock(other.state)


def _paths(folder):
    folder.mkdir()
    contract_path, examples_path = folder / "features.yaml", folder / "examples.json"
    contract_path.write_text(json.dumps(CONTRACT), encoding="utf-8")
    examples_path.write_text(json.dumps(EXAMPLES), encoding="utf-8")
    return contract_path, examples_path


# --- résultat de train_and_promote -----------------------------------------------


class ResultObject:
    def __init__(self, values):
        self.__dict__.update(values)


@pytest.mark.parametrize(
    "result",
    [
        PROMOTED,
        {**PROMOTED, "preset": "rapide", "duration_seconds": 12.5},  # informations en plus ignorées
        ResultObject(PROMOTED),
        schemas.TrainResponse.model_validate(PROMOTED),
    ],
    ids=["dict", "dict-extra", "objet", "TrainResponse"],
)
def test_accepted_result_shapes(app, client, result):
    use(app, FakeTrainer(result=result))
    response = post(client)
    assert response.status_code == 200
    assert response.json() == schemas.TrainResponse.model_validate(PROMOTED).model_dump(mode="json")


@pytest.mark.parametrize(
    ("result", "fields"),
    [
        ({**PROMOTED, "model_version": None}, "(ensemble)"),
        ({**PROMOTED, "decision_threshold": 1.5}, "decision_threshold"),
        ({key: value for key, value in PROMOTED.items() if key != "run_id"}, "run_id"),
        (None, "status"),
        ("promoted", "status"),
    ],
)
def test_a_non_conforming_result_answers_500_without_its_values(app, client, caplog, result, fields):
    use(app, FakeTrainer(result=result))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        error = assert_error(post(client), 500, schemas.ErrorCode.INTERNAL_ERROR)
    assert "contrat" in error.detail
    assert "non conforme à TrainResponse" in caplog.text and fields in caplog.text
    assert PROMOTED["run_id"] not in caplog.text
    use(app, FakeTrainer())
    assert post(client).status_code == 200


@pytest.mark.parametrize("field", ["metric_value", "promotion_threshold", "decision_threshold"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_number_answers_500(app, client, caplog, field, value):
    use(app, FakeTrainer(result={**REJECTED, field: value}))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert_error(post(client), 500, schemas.ErrorCode.INTERNAL_ERROR)
    assert field in caplog.text


class CustomBase(BaseException):
    pass


@pytest.mark.parametrize("error", [SystemExit, KeyboardInterrupt, GeneratorExit, CustomBase])
def test_a_base_exception_in_train_and_promote_stays_in_the_common_format(app, client, caplog, error):
    app.state.reload_model = reloader = Reloader(app)
    use(app, FakeTrainer(error=error("arrêt : /chemin/secret")))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert_error(post(client), 500, schemas.ErrorCode.INTERNAL_ERROR)
    assert f"train_and_promote a levé {error.__name__}" in caplog.text
    assert "/chemin/secret" not in caplog.text
    assert reloader.calls == 1
    use(app, FakeTrainer())
    assert post(client).status_code == 200


def test_a_newline_in_run_id_cannot_forge_a_log_line(app, client, caplog):
    use(app, FakeTrainer(result={**REJECTED, "run_id": "abc\n2026-01-01 INFO faux"}))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert post(client).status_code == 200
    assert not any(line.startswith("2026-01-01 INFO faux") for line in caplog.text.splitlines())


def test_an_async_train_and_promote_is_refused(app, client, caplog):
    async def asynchronous(preset):
        return PROMOTED

    use(app, asynchronous)
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert_error(post(client), 500, schemas.ErrorCode.INTERNAL_ERROR)
    assert "synchrone" in caplog.text


# --- rechargement du modèle servi (raccord avec A2) -----------------------------


class Reloader:
    """Imite app.state.reload_model de A2 : recharge le @champion et met à jour model_info."""

    def __init__(self, app, version="4", error=None):
        self.app, self.version, self.error, self.calls = app, version, error, 0

    def __call__(self):
        self.calls += 1
        if self.error is not None:
            raise self.error
        self.app.state.model_info = schemas.ModelInfo(
            name="credit-default", version=self.version, decision_threshold=0.29
        )


def test_a_promotion_reloads_the_served_model(app, client):
    app.state.model_info = schemas.ModelInfo(name="credit-default", version="3", decision_threshold=0.31)
    app.state.reload_model = reloader = Reloader(app)
    use(app, FakeTrainer(result=PROMOTED))
    assert post(client).json()["model_version"] == "4"
    assert reloader.calls == 1
    assert client.get("/health").json()["model"]["version"] == "4"


def test_a_rejection_does_not_reload(app, client):
    app.state.reload_model = reloader = Reloader(app)
    use(app, FakeTrainer(result=REJECTED))
    assert post(client).status_code == 200
    assert reloader.calls == 0


def test_a_failed_training_reloads_as_a_precaution(app, client):
    app.state.reload_model = reloader = Reloader(app)
    use(app, FakeTrainer(error=RuntimeError("échec")))
    assert post(client).status_code == 500
    assert reloader.calls == 1


def test_without_reload_model_the_promotion_still_answers_200(app, client, caplog):
    # Le lifespan du routeur predict (A2) pose reload_model : on le retire pour ce cas.
    del app.state.reload_model
    use(app, FakeTrainer(result=PROMOTED))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert post(client).status_code == 200
    assert "app.state.reload_model n'est pas défini" in caplog.text


@pytest.mark.parametrize("reload_model", ["pas une fonction", "asynchrone"])
def test_an_unusable_reload_model_is_reported(app, client, caplog, reload_model):
    if reload_model == "asynchrone":

        async def reload_model():
            return None

    app.state.reload_model = reload_model
    use(app, FakeTrainer(result=PROMOTED))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert post(client).status_code == 200
    assert "app.state.reload_model" in caplog.text
    assert "RuntimeWarning" not in caplog.text


@pytest.mark.parametrize("error", [SystemExit, KeyboardInterrupt, CustomBase])
def test_a_base_exception_in_reload_model_keeps_the_200(app, client, caplog, error):
    app.state.reload_model = Reloader(app, error=error("/chemin/secret"))
    use(app, FakeTrainer(result=PROMOTED))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert post(client).status_code == 200
    assert f"Rechargement du modèle servi en échec ({error.__name__})" in caplog.text
    assert "/chemin/secret" not in caplog.text
    assert post(client).status_code == 200


def test_a_failing_reload_keeps_the_200_and_releases_the_lock(app, client, caplog):
    app.state.reload_model = reloader = Reloader(app, error=OSError("artefact illisible : /secret"))
    use(app, FakeTrainer(result=PROMOTED))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert post(client).status_code == 200
        assert post(client).status_code == 200
    assert reloader.calls == 2
    assert "Rechargement du modèle servi en échec (OSError)" in caplog.text
    assert "/secret" not in caplog.text


def test_the_reload_happens_before_the_lock_is_released(app, client):
    seen = []
    app.state.reload_model = lambda: seen.append(train.training_lock(app.state).locked())
    use(app, FakeTrainer(result=PROMOTED))
    assert post(client).status_code == 200
    assert seen == [True]
    assert not train.training_lock(app.state).locked()


def test_the_outcome_is_logged_without_the_key(app, client, caplog):
    use(app, FakeTrainer(result=REJECTED))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        post(client, json={"preset": "faible"})
    assert "Entraînement lancé (préréglage faible)." in caplog.text
    assert "modèle refusé, @champion inchangé" in caplog.text
    assert KEY not in caplog.text


# --- tests ajoutés après l'injection d'erreurs (boucle 2) ---------------------------


def test_the_lock_is_created_once_under_concurrent_first_calls(app, monkeypatch):
    # Sans _LOCKS_GUARD, plusieurs fils verraient training_lock absent et créeraient chacun
    # leur verrou. La création ralentie élargit la fenêtre de concurrence.
    count = 8
    barrier = threading.Barrier(count)
    real_lock = threading.Lock

    def slow_lock():
        lock = real_lock()
        time.sleep(0.05)
        return lock

    def first_call():
        barrier.wait(WAIT_SECONDS)
        return train.training_lock(app.state)

    with ThreadPoolExecutor(max_workers=count) as pool, monkeypatch.context() as patch:
        patch.setattr(train.threading, "Lock", slow_lock)
        locks = [future.result(timeout=WAIT_SECONDS) for future in [pool.submit(first_call) for _ in range(count)]]
    assert len({id(lock) for lock in locks}) == 1
    assert app.state.training_lock is locks[0]


def test_importing_the_api_does_not_import_r2():
    # Dans un processus neuf : la collecte des tests a déjà importé src.api.train ici.
    code = "\n".join(  # noqa: FLY002 - une ligne de code par élément, plus lisible
        [
            "import sys",
            "seen = []",
            "class Watch:",
            "    def find_spec(self, name, path=None, target=None):",
            "        if name == 'src.registry.promotion':",
            "            seen.append(name)",
            "sys.meta_path.insert(0, Watch())",
            "import src.api.main",
            "sys.exit(1 if seen else 0)",
        ]
    )
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, "src.registry.promotion importé au chargement de l'API"


def test_a_non_finite_result_is_logged_without_its_values(app, client, caplog):
    use(app, FakeTrainer(result={**REJECTED, "metric_value": float("nan")}))
    with caplog.at_level(logging.INFO, logger="neuraxion"):
        assert_error(post(client), 500, schemas.ErrorCode.INTERNAL_ERROR)
    assert "metric_value non fini" in caplog.text
    assert REJECTED["run_id"] not in caplog.text


def test_an_unknown_body_field_is_refused_before_training(app, client):
    trainer = use(app, FakeTrainer())
    assert_error(post(client, json={"preset": "rapide", "epochs": 3}), 422, schemas.ErrorCode.VALIDATION_ERROR)
    assert trainer.presets == []


def test_an_empty_key_is_refused_with_401_in_a_direct_call():
    # À travers HTTP, APIKeyHeader convertit déjà un en-tête vide en None.
    request = types.SimpleNamespace(headers=types.SimpleNamespace(getlist=lambda name: [""]))
    with pytest.raises(security.ApiError) as raised:
        security.require_api_key(request, "")
    assert raised.value.status_code == 401


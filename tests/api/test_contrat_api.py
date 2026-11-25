"""Forme du contrat C4 : modèles Pydantic, exemples de docs/api.md et routes encore vides.

Ces tests ne dépendent ni de PostgreSQL ni de MLflow. Ils restent valables quand A1, A2,
A3 et F1 remplacent les corps des routes, sauf ceux qui attendent la réponse 501.
"""

import json
import re
from pathlib import Path

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api import feedback, predict, schemas, train

DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "api.md"
EXAMPLE_PATTERN = re.compile(r"<!-- exemple: (\w+) -->\s*```json\s*(.*?)```", re.DOTALL)
CONTRACT_PATTERN = re.compile(r"<!-- contrat-illustration -->\s*```yaml\s*(.*?)```", re.DOTALL)
REQUEST_PATTERN = re.compile(r"<!-- exemple-requete: predict -->\s*```json\s*(.*?)```", re.DOTALL)
VALID_REQUEST = {"loan_amnt": 12000, "annual_inc": 55000, "home_ownership": "RENT"}


def _doc_examples() -> list[tuple[str, dict]]:
    text = DOC_PATH.read_text(encoding="utf-8")
    return [(name, json.loads(body)) for name, body in EXAMPLE_PATTERN.findall(text)]


def _doc_contract() -> dict:
    (block,) = CONTRACT_PATTERN.findall(DOC_PATH.read_text(encoding="utf-8"))
    return yaml.safe_load(block)


@pytest.fixture
def client(small_contract) -> TestClient:
    app = FastAPI()
    app.include_router(predict.build_router(schemas.build_request_model(small_contract)))
    app.include_router(train.router)
    app.include_router(feedback.router)
    return TestClient(app)


@pytest.fixture
def small_contract() -> dict:
    return {
        "version": "test",
        "features": [
            {"name": "loan_amnt", "type": "numeric", "min": 1000, "max": 40000, "label": "Montant"},
            {"name": "annual_inc", "type": "numeric", "min": 0},
            {"name": "open_acc", "type": "integer", "min": 0, "required": False},
            {"name": "home_ownership", "type": "categorical", "allowed": ["RENT", "OWN", "MORTGAGE"]},
        ],
    }


def test_doc_contains_an_example_for_each_contract_model():
    documented = {name for name, _ in _doc_examples()}
    expected = {
        "ErrorResponse",
        "PredictResponse",
        "TrainRequest",
        "TrainResponse",
        "HealthResponse",
        "FeedbackRequest",
        "FeedbackResponse",
        "SchemaResponse",
    }
    assert expected <= documented


@pytest.mark.parametrize(("model_name", "example"), _doc_examples())
def test_doc_examples_are_valid(model_name, example):
    getattr(schemas, model_name).model_validate(example)


def test_models_refuse_unknown_fields():
    with pytest.raises(ValidationError):
        schemas.TrainRequest.model_validate({"preset": "rapide", "data_path": "/etc"})


def test_train_request_defaults_to_default_preset_and_refuses_others():
    assert schemas.TrainRequest().preset == "default"
    with pytest.raises(ValidationError):
        schemas.TrainRequest(preset="lent")


def test_predict_response_bounds_and_decision_values():
    valid = schemas.PredictResponse.model_config["json_schema_extra"]["examples"][0]
    for field, value in (("default_probability", 1.2), ("decision_threshold", -0.1), ("decision", "maybe")):
        with pytest.raises(ValidationError):
            schemas.PredictResponse.model_validate({**valid, field: value})


@pytest.mark.parametrize(
    ("status", "promoted", "version"),
    [("promoted", False, None), ("rejected", True, "4"), ("promoted", True, None), ("rejected", False, "4")],
)
def test_train_response_refuses_inconsistent_promotion(status, promoted, version):
    valid = schemas.TrainResponse.model_config["json_schema_extra"]["examples"][0]
    with pytest.raises(ValidationError):
        schemas.TrainResponse.model_validate(
            {**valid, "status": status, "promoted": promoted, "model_version": version}
        )


def test_health_status_follows_database_and_model():
    with pytest.raises(ValidationError):
        schemas.HealthResponse(status="ok", database="ok", model=None)
    with pytest.raises(ValidationError):
        schemas.HealthResponse(status="degraded", database="ok", model={"name": "m", "version": "1", "decision_threshold": 0.3})


def test_feedback_is_a_closed_choice():
    with pytest.raises(ValidationError):
        schemas.FeedbackRequest(inference_id="6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d", feedback="le client a payé")


def test_request_model_accepts_a_valid_request(small_contract):
    model = schemas.build_request_model(small_contract)
    request = model(loan_amnt=12000, annual_inc=55000, home_ownership="RENT")
    assert request.open_acc is None


@pytest.mark.parametrize(
    "change",
    [
        {"loan_amnt": 500},
        {"annual_inc": -1},
        {"home_ownership": "BOAT"},
        {"open_acc": 2.5},
        {"nickname": "x"},
    ],
)
def test_request_model_refuses_invalid_values(small_contract, change):
    model = schemas.build_request_model(small_contract)
    base = {"loan_amnt": 12000, "annual_inc": 55000, "home_ownership": "RENT"}
    with pytest.raises(ValidationError):
        model.model_validate({**base, **change})


def test_request_model_requires_mandatory_fields(small_contract):
    model = schemas.build_request_model(small_contract)
    with pytest.raises(ValidationError):
        model.model_validate({"loan_amnt": 12000, "home_ownership": "RENT"})


@pytest.mark.parametrize(
    "contract",
    [
        {"features": []},
        {"features": [{"name": "x", "type": "text"}]},
        {"features": [{"name": "x", "type": "numeric"}, {"name": "x", "type": "numeric"}]},
    ],
)
def test_request_model_refuses_broken_contracts(contract):
    with pytest.raises(ValueError):
        schemas.build_request_model(contract)


@pytest.mark.parametrize(
    ("route", "body", "task"),
    [
        ("/predict", VALID_REQUEST, "A2"),
        ("/train", {"preset": "rapide"}, "A3"),
        ("/feedback", {"inference_id": "6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d", "feedback": "agree"}, "F1"),
    ],
)
def test_empty_routes_answer_501_in_the_common_format(client, route, body, task):
    response = client.post(route, json=body)
    assert response.status_code == 501
    error = schemas.ErrorResponse.model_validate(response.json())
    assert error.code == schemas.ErrorCode.NOT_IMPLEMENTED
    assert task in error.detail


def test_routes_are_documented_in_openapi(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/predict", "/train", "/feedback"} <= set(paths)
    assert "PredictResponse" in json.dumps(paths["/predict"])


def _predict_example() -> dict:
    return dict(schemas.PredictResponse.model_config["json_schema_extra"]["examples"][0])


@pytest.mark.parametrize(
    ("field", "value"),
    [("default_probability", -0.01), ("decision_threshold", 1.01), ("model_version", "@champion"), ("model_version", "champion")],
)
def test_predict_response_refuses_out_of_range_values_and_aliases(field, value):
    with pytest.raises(ValidationError, match=field):
        schemas.PredictResponse.model_validate({**_predict_example(), field: value})


@pytest.mark.parametrize(
    ("probability", "threshold", "decision", "valid"),
    [
        (0.31, 0.31, "reject", True),
        (0.30, 0.31, "approve", True),
        (0.31, 0.31, "approve", False),
        (0.23, 0.31, "reject", False),
    ],
)
def test_predict_response_decision_follows_threshold(probability, threshold, decision, valid):
    data = {**_predict_example(), "default_probability": probability, "decision_threshold": threshold, "decision": decision}
    if valid:
        assert schemas.PredictResponse.model_validate(data).decision == decision
    else:
        with pytest.raises(ValidationError):
            schemas.PredictResponse.model_validate(data)


@pytest.mark.parametrize("preset", ["default", "rapide", "faible"])
def test_train_request_accepts_each_preset_of_the_protocol(preset):
    assert schemas.TrainRequest(preset=preset).preset == preset


def test_health_is_degraded_when_database_is_unavailable():
    model = {"name": "credit-default", "version": "1", "decision_threshold": 0.3}
    assert schemas.HealthResponse(status="degraded", database="unavailable", model=model).status == "degraded"
    with pytest.raises(ValidationError):
        schemas.HealthResponse(status="ok", database="unavailable", model=model)


def test_field_errors_are_only_for_validation_errors():
    with pytest.raises(ValidationError):
        schemas.ErrorResponse(detail="x", code="MODEL_UNAVAILABLE", errors=[{"field": "a", "message": "b"}])


def test_request_model_refuses_values_above_maximum(small_contract):
    model = schemas.build_request_model(small_contract)
    with pytest.raises(ValidationError, match="loan_amnt"):
        model(loan_amnt=40001, annual_inc=55000, home_ownership="RENT")


def test_predict_requires_a_body(client):
    assert client.post("/predict").status_code == 422


@pytest.mark.parametrize("body", [None, {}])
def test_train_without_preset_uses_default_and_reaches_the_stub(client, body):
    response = client.post("/train") if body is None else client.post("/train", json=body)
    assert response.status_code == 501


def test_wire_codes_are_the_documented_strings(client):
    # Valeur littérale, indépendante de l'énumération : un client se fie à cette chaîne.
    assert client.post("/predict", json=VALID_REQUEST).json()["code"] == "NOT_IMPLEMENTED"


def test_openapi_announces_the_documented_error_statuses(client):
    paths = client.get("/openapi.json").json()["paths"]
    expected = {"/predict": {"422", "501", "503"}, "/train": {"401", "403", "409", "422", "501"}, "/feedback": {"404", "422", "501"}}
    for path, statuses in expected.items():
        assert statuses <= set(paths[path]["post"]["responses"])


def test_every_json_block_of_the_doc_parses():
    text = DOC_PATH.read_text(encoding="utf-8")
    blocks = re.findall(r"^```json[^\S\n]*\n(.*?)^```", text, re.MULTILINE | re.DOTALL)
    assert len(blocks) >= 10
    for block in blocks:
        json.loads(block)


@pytest.mark.parametrize(
    "body",
    [{**VALID_REQUEST, "nickname": "x"}, {**VALID_REQUEST, "loan_amnt": 500}, {"loan_amnt": 12000}],
)
def test_predict_route_validates_with_the_contract_model(client, body):
    assert client.post("/predict", json=body).status_code == 422


def test_openapi_describes_the_contract_request(client):
    spec = client.get("/openapi.json").json()
    request_ref = spec["paths"]["/predict"]["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    request_schema = spec["components"]["schemas"][request_ref.rsplit("/", 1)[-1]]
    assert set(request_schema["properties"]) == {"loan_amnt", "annual_inc", "open_acc", "home_ownership"}


def test_doc_schema_example_is_produced_by_the_request_factory():
    documented = dict(_doc_examples())["SchemaResponse"]
    model = schemas.build_request_model(_doc_contract())
    assert documented["request_schema"] == model.model_json_schema()
    assert len(documented["examples"]) == 3
    for example in documented["examples"].values():
        model.model_validate(example)


def test_doc_predict_request_is_valid_for_the_illustration_contract():
    (block,) = REQUEST_PATTERN.findall(DOC_PATH.read_text(encoding="utf-8"))
    schemas.build_request_model(_doc_contract()).model_validate(json.loads(block))


def test_predict_response_requires_a_timezone():
    with pytest.raises(ValidationError):
        schemas.PredictResponse.model_validate({**_predict_example(), "created_at": "2026-11-27T10:15:42"})

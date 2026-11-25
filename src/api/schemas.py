"""Contrat de l'API (C4) : modèles Pydantic des requêtes, des réponses et des erreurs.

Le contrat est expliqué, avec un exemple par route, dans docs/api.md. Les noms des champs
suivent le plan de l'équipe : promotion_threshold et decision_threshold restent distincts.
La requête de POST /predict n'est pas écrite ici à la main : build_request_model() la
construit à partir du contrat de données (contracts/features.yaml, lot P1).
"""

from enum import StrEnum
from typing import Any, Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, create_model, model_validator

Decision = Literal["approve", "reject"]
Preset = Literal["default", "rapide", "faible"]
FeedbackValue = Literal["agree", "disagree"]

FEATURE_TYPES = ("numeric", "integer", "categorical")

# Une version du Model Registry est un entier positif écrit en texte ("1", "2"...), jamais l'alias.
MODEL_VERSION_PATTERN = r"^[1-9][0-9]*$"


class ErrorCode(StrEnum):
    """Codes machine du format d'erreur commun, avec le statut HTTP qui les accompagne."""

    VALIDATION_ERROR = "VALIDATION_ERROR"  # 422
    UNAUTHORIZED = "UNAUTHORIZED"  # 401
    FORBIDDEN = "FORBIDDEN"  # 403
    NOT_FOUND = "NOT_FOUND"  # 404
    TRAINING_IN_PROGRESS = "TRAINING_IN_PROGRESS"  # 409
    INTERNAL_ERROR = "INTERNAL_ERROR"  # 500
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"  # 501
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"  # 503
    DATABASE_UNAVAILABLE = "DATABASE_UNAVAILABLE"  # 503


class _Contract(BaseModel):
    """Base des modèles du contrat : tout champ inconnu est refusé."""

    model_config = ConfigDict(extra="forbid")


class FieldError(_Contract):
    """Un champ refusé par la validation et la raison du refus."""

    field: str
    message: str


class ErrorResponse(_Contract):
    """Format commun à toutes les erreurs de l'API."""

    detail: str = Field(description="Message compréhensible par une personne.")
    code: ErrorCode = Field(description="Code stable, à tester côté client plutôt que le message.")
    errors: list[FieldError] | None = Field(
        default=None, description="Champs refusés, seulement pour VALIDATION_ERROR."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detail": "Aucun modèle validé n'est disponible. Lancez POST /train ou consultez les logs du backend.",
                    "code": "MODEL_UNAVAILABLE",
                    "errors": None,
                }
            ]
        }
    )

    @model_validator(mode="after")
    def _field_errors_only_for_validation(self) -> "ErrorResponse":
        if self.errors is not None and self.code != ErrorCode.VALIDATION_ERROR:
            raise ValueError("errors n'est rempli que pour VALIDATION_ERROR.")
        return self

    @classmethod
    def not_implemented(cls, route: str, task: str) -> "ErrorResponse":
        """Réponse des routes encore vides : elle nomme la tâche qui les remplira."""
        return cls(
            detail=f"{route} n'est pas encore implémenté (tâche {task}).",
            code=ErrorCode.NOT_IMPLEMENTED,
        )


class PredictResponse(_Contract):
    """Réponse de POST /predict : la prédiction et de quoi retrouver l'inférence en base."""

    inference_id: UUID = Field(description="Identifiant de la ligne créée dans la table inferences.")
    created_at: AwareDatetime = Field(description="Horodatage de la réception de la requête, avec fuseau.")
    default_probability: float = Field(ge=0, le=1, description="Probabilité de défaut du prêt.")
    decision: Decision = Field(description="reject si default_probability >= decision_threshold.")
    decision_threshold: float = Field(ge=0, le=1, description="Seuil de décision du modèle servi.")
    model_name: str = Field(description="Modèle enregistré dans MLflow, par exemple credit-default.")
    model_version: str = Field(
        pattern=MODEL_VERSION_PATTERN, description="Version résolue depuis l'alias @champion."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inference_id": "6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d",
                    "created_at": "2026-11-27T10:15:42.120000+01:00",
                    "default_probability": 0.23,
                    "decision": "approve",
                    "decision_threshold": 0.31,
                    "model_name": "credit-default",
                    "model_version": "3",
                }
            ]
        }
    )

    @model_validator(mode="after")
    def _decision_follows_threshold(self) -> "PredictResponse":
        expected = "reject" if self.default_probability >= self.decision_threshold else "approve"
        if self.decision != expected:
            raise ValueError(f"decision doit valoir {expected} pour cette probabilité et ce seuil.")
        return self


class TrainRequest(_Contract):
    """Corps de POST /train : le préréglage d'entraînement (config/training.yaml)."""

    preset: Preset = Field(default="default", description="default, rapide ou faible.")

    model_config = ConfigDict(json_schema_extra={"examples": [{"preset": "rapide"}]})


class TrainResponse(_Contract):
    """Réponse de POST /train : le run MLflow, sa métrique et la décision de promotion."""

    status: Literal["promoted", "rejected"] = Field(description="Issue de l'entraînement.")
    run_id: str = Field(description="Identifiant du run MLflow.")
    metric_name: str = Field(description="Métrique de promotion, roc_auc dans le protocole.")
    metric_value: float = Field(description="Valeur obtenue sur la validation.")
    promotion_threshold: float = Field(description="Seuil de promotion lu dans config/training.yaml.")
    promoted: bool = Field(description="Vrai si le modèle est devenu @champion.")
    model_version: str | None = Field(
        pattern=MODEL_VERSION_PATTERN, description="Nouvelle version si promue, sinon null."
    )
    decision_threshold: float = Field(ge=0, le=1, description="Seuil de décision choisi sur la validation.")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "promoted",
                    "run_id": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
                    "metric_name": "roc_auc",
                    "metric_value": 0.71,
                    "promotion_threshold": 0.65,
                    "promoted": True,
                    "model_version": "4",
                    "decision_threshold": 0.29,
                },
                {
                    "status": "rejected",
                    "run_id": "0f9e8d7c6b5a49382716f5e4d3c2b1a0",
                    "metric_name": "roc_auc",
                    "metric_value": 0.58,
                    "promotion_threshold": 0.65,
                    "promoted": False,
                    "model_version": None,
                    "decision_threshold": 0.5,
                },
            ]
        }
    )

    @model_validator(mode="after")
    def _promotion_is_consistent(self) -> "TrainResponse":
        if self.promoted != (self.status == "promoted"):
            raise ValueError("status et promoted doivent concorder.")
        if self.promoted != (self.model_version is not None):
            raise ValueError("model_version est renseigné si et seulement si le modèle est promu.")
        return self


class ModelInfo(_Contract):
    """Modèle actuellement servi."""

    name: str
    version: str = Field(pattern=MODEL_VERSION_PATTERN)
    decision_threshold: float = Field(ge=0, le=1)


class HealthResponse(_Contract):
    """Réponse de GET /health. model vaut null tant qu'aucun modèle n'est promu."""

    status: Literal["ok", "degraded"] = Field(description="degraded si la base ou le modèle manque.")
    database: Literal["ok", "unavailable"]
    model: ModelInfo | None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "database": "ok",
                    "model": {"name": "credit-default", "version": "3", "decision_threshold": 0.31},
                },
                {"status": "degraded", "database": "ok", "model": None},
            ]
        }
    )

    @model_validator(mode="after")
    def _status_matches_dependencies(self) -> "HealthResponse":
        healthy = self.database == "ok" and self.model is not None
        if (self.status == "ok") != healthy:
            raise ValueError("status vaut ok si et seulement si la base répond et un modèle est chargé.")
        return self


class SchemaResponse(_Contract):
    """Réponse de GET /schema (tâche A1) : de quoi générer le formulaire de l'interface."""

    contract_version: str = Field(description="Version du contrat de données (contracts/features.yaml).")
    request_schema: dict[str, Any] = Field(
        description="JSON Schema de la requête de POST /predict, produit par build_request_model()."
    )
    examples: dict[str, dict[str, Any]] = Field(
        description="Demandes complètes et valides (contracts/examples.json), par nom de profil."
    )


class FeedbackRequest(_Contract):
    """Corps de POST /feedback (facultatif, tâche F1) : un choix fermé, sans texte libre."""

    inference_id: UUID
    feedback: FeedbackValue

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [{"inference_id": "6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d", "feedback": "disagree"}]
        }
    )


class FeedbackResponse(_Contract):
    """Réponse de POST /feedback : l'inférence complétée."""

    inference_id: UUID
    feedback: FeedbackValue


def build_request_model(contract: dict[str, Any], name: str = "PredictRequest") -> type[BaseModel]:
    """Construit le modèle de requête de POST /predict à partir du contrat de données.

    Format attendu pour chaque entrée de contract["features"] (hypothèse à confirmer avec P1
    à la revue C6) : name et type (numeric, integer ou categorical) obligatoires ; min, max,
    allowed (valeurs permises), required (vrai par défaut), label, unit et example facultatifs.
    Un champ facultatif absent vaut None. Le modèle produit refuse les champs inconnus.

    Paramètres
        contract : contrat déjà lu, par exemple avec src.training.load_contract().
        name : nom du modèle produit, visible dans le schéma OpenAPI.

    Renvoie
        Une classe Pydantic, à utiliser comme corps de la route.

    Lève
        ValueError si le contrat est vide, si un nom est répété ou si un type est inconnu.
    """
    features = contract.get("features")
    if not features:
        raise ValueError("Le contrat ne déclare aucune variable (clé features).")

    fields: dict[str, Any] = {}
    for feature in features:
        feature_name = feature["name"]
        if feature_name in fields:
            raise ValueError(f"Variable déclarée deux fois dans le contrat : {feature_name}.")
        fields[feature_name] = _field_definition(feature)

    return create_model(name, __config__=ConfigDict(extra="forbid"), **fields)


def _field_definition(feature: dict[str, Any]) -> tuple[Any, Any]:
    """Type Python et contraintes Pydantic d'une variable du contrat."""
    kind = feature["type"]
    if kind not in FEATURE_TYPES:
        raise ValueError(f"Type inconnu pour {feature['name']} : {kind} (attendu : {', '.join(FEATURE_TYPES)}).")

    allowed = feature.get("allowed")
    if allowed:
        annotation: Any = Literal[tuple(allowed)]
    elif kind == "categorical":
        annotation = str
    elif kind == "integer":
        annotation = int
    else:
        annotation = float

    constraints: dict[str, Any] = {}
    if kind != "categorical":
        if feature.get("min") is not None:
            constraints["ge"] = feature["min"]
        if feature.get("max") is not None:
            constraints["le"] = feature["max"]
    if feature.get("label"):
        constraints["title"] = feature["label"]
    if feature.get("unit"):
        constraints["description"] = f"Unité : {feature['unit']}"
    if feature.get("example") is not None:
        constraints["examples"] = [feature["example"]]

    if feature.get("required", True):
        return annotation, Field(..., **constraints)
    return annotation | None, Field(default=None, **constraints)

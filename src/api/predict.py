"""Route POST /predict. Contrat posé par C4, logique écrite par la tâche A2.

Le corps de la requête dépend du contrat de données : A1 construit sa classe au démarrage
avec build_request_model(), puis crée le routeur avec build_router(). La même classe sert
ainsi à la validation (422 sur un champ inconnu), à OpenAPI et à GET /schema.
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .schemas import ErrorResponse, PredictResponse


def build_router(request_model: type[BaseModel]) -> APIRouter:
    """Routeur de POST /predict dont le corps est validé par request_model."""
    router = APIRouter(tags=["inférence"])

    @router.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            422: {"model": ErrorResponse, "description": "Entrée refusée par le contrat de données."},
            501: {"model": ErrorResponse, "description": "Route pas encore implémentée."},
            503: {"model": ErrorResponse, "description": "Aucun modèle validé, ou base indisponible."},
        },
        summary="Prédire le risque de défaut d'une demande de prêt",
    )
    def predict(payload: request_model) -> JSONResponse:  # type: ignore[valid-type]
        error = ErrorResponse.not_implemented("POST /predict", "A2")
        return JSONResponse(status_code=501, content=error.model_dump(mode="json"))

    return router

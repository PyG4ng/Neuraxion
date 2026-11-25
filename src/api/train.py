"""Route POST /train. Contrat posé par C4, logique et clé API écrites par la tâche A3."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .schemas import ErrorResponse, TrainRequest, TrainResponse

router = APIRouter(tags=["entraînement"])


@router.post(
    "/train",
    response_model=TrainResponse,
    responses={
        401: {"model": ErrorResponse, "description": "En-tête X-API-Key absent."},
        403: {"model": ErrorResponse, "description": "Clé fausse, ou clé non configurée sur le serveur."},
        409: {"model": ErrorResponse, "description": "Un entraînement est déjà en cours."},
        422: {"model": ErrorResponse, "description": "Préréglage inconnu."},
        501: {"model": ErrorResponse, "description": "Route pas encore implémentée."},
    },
    summary="Entraîner un modèle et le promouvoir s'il atteint le seuil",
)
def train(request: TrainRequest | None = None) -> JSONResponse:
    """Lancera train_and_promote(request.preset) derrière la clé X-API-Key (tâche A3).

    Le corps est facultatif : sans corps, le préréglage est default.
    """
    request = request or TrainRequest()
    error = ErrorResponse.not_implemented("POST /train", "A3")
    return JSONResponse(status_code=501, content=error.model_dump(mode="json"))

"""Route POST /feedback, facultative. Contrat posé par C4, logique écrite par la tâche F1."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .schemas import ErrorResponse, FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["inférence"])


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Inférence inconnue."},
        422: {"model": ErrorResponse, "description": "Valeur de feedback non permise."},
        501: {"model": ErrorResponse, "description": "Route pas encore implémentée."},
    },
    summary="Indiquer si l'on est d'accord avec la décision d'une inférence",
)
def feedback(request: FeedbackRequest) -> JSONResponse:
    """Complétera la ligne de l'inférence dans PostgreSQL (tâche F1)."""
    error = ErrorResponse.not_implemented("POST /feedback", "F1")
    return JSONResponse(status_code=501, content=error.model_dump(mode="json"))

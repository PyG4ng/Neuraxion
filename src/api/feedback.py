"""Route POST /feedback, facultative. Contrat posé par C4, logique écrite par la tâche F1.

Le client indique s'il est d'accord (agree) ou non (disagree) avec la décision d'une
inférence réussie, désignée par l'inference_id que POST /predict lui a renvoyé. Le choix est
enregistré dans la colonne feedback de la table inferences ; un nouvel appel remplace le
précédent. Aucun texte libre n'est accepté (422).

Le feedback n'est jamais utilisé pour l'entraînement (décision d'équipe H14) : l'issue réelle
d'un prêt n'est connue qu'après 36 mois.
"""

import logging
from uuid import UUID

import psycopg
from fastapi import APIRouter

from src.api.errors import ApiError
from src.api.schemas import (
    ErrorCode,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    FeedbackValue,
)
from src.db.connection import DatabaseNotConfigured, connect
from src.db.inferences import save_feedback
from src.db.init_db import create_schema

logger = logging.getLogger("neuraxion.api")

NOT_FOUND_DETAIL = (
    "Aucune inférence réussie ne porte cet identifiant. "
    "Utilisez l'inference_id renvoyé par une réponse 200 de POST /predict."
)
DATABASE_UNAVAILABLE_DETAIL = (
    "La base de données ne répond pas : le feedback n'a pas été enregistré. Réessayez dans quelques instants."
)

# PostgreSQL injoignable ou non configurée : le feedback ne peut pas être enregistré.
DATABASE_DOWN = (
    DatabaseNotConfigured,
    psycopg.OperationalError,
    psycopg.InterfaceError,
)

router = APIRouter(tags=["inférence"])


def store_feedback(inference_id: UUID, value: FeedbackValue) -> bool:
    """Enregistre le feedback ; renvoie False si aucune inférence réussie ne porte cet id.

    Si la table ou la colonne feedback manque (base créée avant F1 et absente au démarrage,
    table supprimée depuis), le schéma est complété puis l'écriture est retentée une fois.
    Compléter le schéma demande d'être propriétaire de la table : avec les seuls droits de la
    route, create_schema() lève InsufficientPrivilege (réponse 500).

    Lève DatabaseNotConfigured ou psycopg.Error.
    """
    with connect() as connection:
        try:
            return save_feedback(inference_id, value, connection)
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
            connection.rollback()
            create_schema(connection)
            return save_feedback(inference_id, value, connection)


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Inférence inconnue, ou inférence en erreur (sans décision).",
        },
        422: {
            "model": ErrorResponse,
            "description": "Valeur de feedback non permise, ou identifiant mal formé.",
        },
        503: {"model": ErrorResponse, "description": "Base de données indisponible."},
    },
    summary="Indiquer si l'on est d'accord avec la décision d'une inférence",
)
def feedback(payload: FeedbackRequest) -> FeedbackResponse:
    """Complète la ligne de l'inférence dans PostgreSQL avec agree ou disagree."""
    try:
        found = store_feedback(payload.inference_id, payload.feedback)
    except DATABASE_DOWN as error:
        logger.error(
            "Feedback de l'inférence %s non enregistré : PostgreSQL injoignable ou non configurée (%s). Réponse 503.",
            payload.inference_id,
            type(error).__name__,
        )
        raise ApiError(503, ErrorCode.DATABASE_UNAVAILABLE, DATABASE_UNAVAILABLE_DETAIL) from None
    if not found:
        logger.info("Feedback refusé : aucune inférence réussie %s.", payload.inference_id)
        raise ApiError(404, ErrorCode.NOT_FOUND, NOT_FOUND_DETAIL)
    # La valeur choisie n'est pas journalisée, comme la décision dans POST /predict.
    logger.info("Feedback enregistré pour l'inférence %s.", payload.inference_id)
    return FeedbackResponse(inference_id=payload.inference_id, feedback=payload.feedback)

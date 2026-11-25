"""Route POST /predict. Contrat posé par C4, logique écrite par la tâche A2.

Le corps de la requête dépend du contrat de données : A1 construit sa classe au démarrage
avec build_request_model(), puis crée le routeur avec build_router(). La même classe sert
ainsi à la validation (422 sur un champ inconnu), à OpenAPI et à GET /schema.

Au démarrage (lifespan du routeur, que FastAPI fusionne avec celui de l'application) :

1. create_schema() crée la table inferences si elle n'existe pas. Une base absente
   n'empêche pas le démarrage : la création est retentée à la première inférence ;
2. le @champion est chargé une fois (served_model.py) et gardé en mémoire ;
3. app.state.model_info décrit le modèle servi (lu par /health) et app.state.reload_model,
   une fonction sans argument, recharge le @champion (appelée par A3 après une promotion).

Chaque inférence validée est enregistrée dans inferences, réussie ou en erreur, avec sa
durée. Ni la réponse ni les logs ne recopient les données de la demande.
"""

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import psycopg
from fastapi import APIRouter, FastAPI, Request
from pydantic import BaseModel

from src.api import served_model
from src.api.errors import ApiError
from src.api.schemas import ErrorCode, ErrorResponse, ModelInfo, PredictResponse
from src.db.connection import DatabaseNotConfigured, connect
from src.db.inferences import InferenceRecord, save_inference
from src.db.init_db import create_schema

logger = logging.getLogger("neuraxion.api")

# Message prévu par le contrat (exemple de ErrorResponse, docs/api.md).
MODEL_UNAVAILABLE_DETAIL = "Aucun modèle validé n'est disponible. Lancez POST /train ou consultez les logs du backend."
DATABASE_UNAVAILABLE_DETAIL = (
    "La base de données ne répond pas : l'inférence n'a pas été enregistrée. Réessayez dans quelques instants."
)
MODEL_FAILURE_DETAIL = "Le modèle n'a pas pu calculer la prédiction. Le détail est dans les logs du backend."

# Causes enregistrées dans error_message : des textes fixes, jamais un message d'exception.
NO_MODEL_CAUSE = "Aucun modèle validé n'est disponible."
MODEL_OUTPUT_CAUSE = "Sortie du modèle inexploitable : {reason}"
MODEL_FAILURE_CAUSE = "Le modèle a échoué pendant la prédiction ({error_type})."

# PostgreSQL injoignable ou non configurée : l'inférence ne peut pas être enregistrée.
DATABASE_DOWN = (DatabaseNotConfigured, psycopg.OperationalError, psycopg.InterfaceError)


def categorical_fields(request_model: type[BaseModel]) -> frozenset[str]:
    """Variables catégorielles du contrat : celles dont le JSON Schema est une chaîne."""
    properties = request_model.model_json_schema().get("properties", {})
    found = set()
    for name, schema in properties.items():
        types = {schema.get("type")} | {branch.get("type") for branch in schema.get("anyOf", [])}
        if "string" in types:
            found.add(name)
    return frozenset(found)


def _milliseconds_since(started: float) -> int:
    return max(0, round((time.perf_counter() - started) * 1000))


def _initialize_schema() -> None:
    """create_schema() au démarrage. Un échec est journalisé, sans empêcher le démarrage."""
    try:
        with connect() as connection:
            create_schema(connection)
    except DatabaseNotConfigured as error:
        logger.warning(
            "Table inferences non créée : base non configurée (%s). "
            "Elle sera créée par la première inférence qui ne la trouve pas.",
            error,
        )
        return
    except psycopg.Error as error:
        logger.warning(
            "Table inferences non créée : PostgreSQL injoignable ou script refusé (%s). "
            "Elle sera créée par la première inférence qui ne la trouve pas.",
            type(error).__name__,
        )
        return
    logger.info("Table inferences prête.")


def store_inference(record: InferenceRecord) -> None:
    """Enregistre l'inférence ; si la table n'existe pas (base absente au démarrage, table
    supprimée depuis), la crée puis réessaie une fois.

    L'insertion est tentée d'abord : un rôle qui a seulement le droit d'insérer dans une
    table déjà créée n'a pas besoin du droit CREATE.

    Lève DatabaseNotConfigured ou psycopg.Error comme save_inference().
    """
    with connect() as connection:
        try:
            save_inference(record, connection)
        except psycopg.errors.UndefinedTable:
            connection.rollback()
            create_schema(connection)
            save_inference(record, connection)


def _store_failed_inference(record: InferenceRecord) -> None:
    """Enregistre une inférence en erreur. Un échec de la base est journalisé, pas propagé :
    la réponse garde la cause première (modèle absent ou en erreur)."""
    try:
        store_inference(record)
    except Exception as error:  # noqa: BLE001 - la cause première de l'erreur reste celle du modèle
        logger.warning(
            "Inférence %s en erreur non enregistrée : base indisponible ou ligne refusée (%s).",
            record.id,
            type(error).__name__,
        )


def _make_reloader(app: FastAPI, features: tuple[str, ...], categorical: frozenset[str]):
    lock = threading.Lock()

    def reload_model() -> ModelInfo | None:
        """Recharge le @champion et met à jour le modèle servi et app.state.model_info.

        Sans champion, ou avec un champion qui ne peut pas être servi (variables différentes
        du contrat, version, seuil ou classes inexploitables), plus aucun modèle n'est servi
        (503). Si le chargement échoue (MLflow injoignable, exception de load_champion()),
        le modèle déjà servi est gardé. Chaque échec est journalisé. Renvoie la description
        du modèle servi après l'appel. Fonction synchrone et bloquante : à appeler hors de la
        boucle d'événements (route def, ou run_in_threadpool depuis une route async).
        """
        with lock:
            served = _load(getattr(app.state, "served_model", None))
            app.state.served_model = served
            app.state.model_info = served.info if served is not None else None
            return app.state.model_info

    def _load(current: served_model.ServedChampion | None) -> served_model.ServedChampion | None:
        """Modèle à servir après le rechargement, avec une ligne de log par issue."""
        try:
            champion = served_model.load_champion_from_registry()
        except Exception as error:  # noqa: BLE001 - MLflow, R1 : l'API reste debout
            # Exception levée par load_champion() lui-même, de quelque type qu'elle soit : panne
            # passagère possible, l'ancien modèle est gardé. Seul le type est journalisé.
            logger.warning("Chargement du @champion impossible (%s). %s", served_model.error_type_name(error), _kept(current))
            return current
        try:
            served = served_model.adapt_champion(champion, features, categorical)
        except served_model.ChampionError as error:
            # Le registre désigne un champion que l'API ne sait pas servir : garder l'ancien
            # servirait une autre version que @champion (garde-fou du plan : 503).
            logger.warning("Le @champion ne peut pas être servi : %s POST /predict répond 503.", error)
            return None
        if served is None:
            logger.warning("Aucun modèle sous l'alias @champion : POST /predict répond 503.")
        else:
            logger.info(
                "Modèle servi : %s version %s, seuil de décision %s.",
                served.name,
                served.version,
                served.decision_threshold,
            )
        return served

    return reload_model


def _kept(current: served_model.ServedChampion | None) -> str:
    if current is None:
        return "Aucun modèle n'est servi : POST /predict répond 503."
    return f"Le modèle déjà servi est gardé ({current.name} version {current.version})."


def build_router(request_model: type[BaseModel]) -> APIRouter:
    """Routeur de POST /predict dont le corps est validé par request_model."""
    features = tuple(request_model.model_fields)
    categorical = categorical_fields(request_model)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.served_model = None
        app.state.model_info = None
        app.state.reload_model = _make_reloader(app, features, categorical)
        # Appels bloquants (PostgreSQL, MLflow) hors de la boucle d'événements.
        await asyncio.to_thread(_initialize_schema)
        await asyncio.to_thread(app.state.reload_model)
        yield

    router = APIRouter(tags=["inférence"], lifespan=lifespan)

    @router.post(
        "/predict",
        response_model=PredictResponse,
        responses={
            422: {"model": ErrorResponse, "description": "Entrée refusée par le contrat de données."},
            500: {"model": ErrorResponse, "description": "Le modèle a échoué ; l'inférence est enregistrée en erreur."},
            503: {"model": ErrorResponse, "description": "Aucun modèle validé, ou base indisponible."},
        },
        summary="Prédire le risque de défaut d'une demande de prêt",
    )
    def predict(payload: request_model, request: Request) -> PredictResponse:  # type: ignore[valid-type]
        started = time.perf_counter()
        created_at = datetime.now(UTC)
        inference_id = uuid4()
        state = request.app.state
        # Seules les variables du contrat (la classe de requête refuse déjà tout autre champ).
        input_data = {name: getattr(payload, name) for name in features}
        served = getattr(state, "served_model", None)

        if served is None:
            _store_failed_inference(
                _error_record(inference_id, created_at, input_data, started, NO_MODEL_CAUSE, None)
            )
            raise ApiError(503, ErrorCode.MODEL_UNAVAILABLE, MODEL_UNAVAILABLE_DETAIL)

        try:
            probability = served.default_probability(input_data)
        except Exception as error:  # noqa: BLE001 - toute erreur du modèle est tracée en base
            # Causes choisies dans une liste fermée : jamais le message d'une exception.
            if isinstance(error, served_model.ModelOutputError) and str(error) in served_model.OUTPUT_PROBLEMS:
                cause = MODEL_OUTPUT_CAUSE.format(reason=error)
                error_type = "sortie inexploitable"
            else:
                if isinstance(error, served_model.ModelFailure):
                    error_type = error.error_type
                else:
                    error_type = served_model.error_type_name(error)
                cause = MODEL_FAILURE_CAUSE.format(error_type=error_type)
            logger.error(
                "Inférence %s : le modèle %s version %s a échoué (%s).",
                inference_id,
                served.name,
                served.version,
                error_type,
            )
            _store_failed_inference(_error_record(inference_id, created_at, input_data, started, cause, served))
            raise ApiError(500, ErrorCode.INTERNAL_ERROR, MODEL_FAILURE_DETAIL) from None

        decision = served.decide(probability)
        latency_ms = _milliseconds_since(started)
        record = InferenceRecord(
            id=inference_id,
            created_at=created_at,
            input_data=input_data,
            status="success",
            latency_ms=latency_ms,
            prediction=probability,
            decision=decision,
            decision_threshold=served.decision_threshold,
            model_name=served.name,
            model_version=served.version,
        )
        try:
            store_inference(record)
        except DATABASE_DOWN as error:
            logger.error(
                "Inférence %s non enregistrée : PostgreSQL injoignable ou non configurée (%s). Réponse 503.",
                inference_id,
                type(error).__name__,
            )
            raise ApiError(503, ErrorCode.DATABASE_UNAVAILABLE, DATABASE_UNAVAILABLE_DETAIL) from None

        logger.info(
            "Inférence %s : modèle %s version %s, %d ms.", inference_id, served.name, served.version, latency_ms
        )
        return PredictResponse(
            inference_id=inference_id,
            created_at=created_at,
            default_probability=probability,
            decision=decision,
            decision_threshold=served.decision_threshold,
            model_name=served.name,
            model_version=served.version,
        )

    return router


def _error_record(
    inference_id: UUID,
    created_at: datetime,
    input_data: dict[str, Any],
    started: float,
    cause: str,
    served: served_model.ServedChampion | None,
) -> InferenceRecord:
    """Ligne status = 'error' : elle garde ce qui est connu du modèle et laisse vide le reste."""
    return InferenceRecord(
        id=inference_id,
        created_at=created_at,
        input_data=input_data,
        status="error",
        latency_ms=_milliseconds_since(started),
        decision_threshold=served.decision_threshold if served is not None else None,
        model_name=served.name if served is not None else None,
        model_version=served.version if served is not None else None,
        error_message=cause,
    )

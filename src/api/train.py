"""Route POST /train (tâche A3) : entraîner un modèle et le promouvoir s'il atteint le seuil.

Contrôles, dans cet ordre :

1. la clé X-API-Key (src/api/security.py) : 401 si elle manque, 403 si elle est fausse ou si
   TRAIN_API_KEY n'est pas configurée ;
2. la fonction train_and_promote(preset) du lot P3 (tâche R2, src/registry/promotion.py) :
   501 tant qu'elle n'est pas livrée ;
3. le corps : 422 pour un préréglage inconnu ;
4. un seul entraînement à la fois par processus : 409 TRAINING_IN_PROGRESS sinon.

La route est synchrone : FastAPI l'exécute dans un fil de son pool, si bien qu'un entraînement
de plusieurs minutes ne bloque pas la boucle d'événements (/health et /predict continuent de
répondre). Le verrou est un threading.Lock pris sans attente et libéré à la fin de
l'entraînement, même si le client s'est déconnecté entre-temps.

Après une promotion, la route appelle app.state.reload_model(), posé par A2 (tâche #15) : une
fonction synchrone sans argument qui recharge le @champion et met à jour app.state.model_info.
Si elle manque, la réponse reste 200 (l'entraînement a réussi) et un avertissement est
journalisé ; si elle échoue, la réponse reste aussi 200 et l'erreur est journalisée.

Une BaseException levée par le code de R2 (sys.exit(), KeyboardInterrupt levée à la main dans
ce fil) est traitée comme une exception ordinaire : réponse au format commun, sans son message
dans les logs.
"""

import importlib
import inspect
import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError
from starlette.datastructures import State

from src.api.errors import ApiError, _app_frames
from src.api.schemas import ErrorCode, ErrorResponse, TrainRequest, TrainResponse
from src.api.security import require_api_key

logger = logging.getLogger("neuraxion.api")

PROMOTION_MODULE = "src.registry.promotion"
PROMOTION_FUNCTION = "train_and_promote"
NOT_DELIVERED_DETAIL = (
    "POST /train n'est pas encore raccordé : train_and_promote(preset) (tâche R2, "
    "src/registry/promotion.py) n'est pas encore livrée."
)
IN_PROGRESS_DETAIL = "Un entraînement est déjà en cours. Réessayez quand il sera terminé."
FAILED_DETAIL = "Erreur interne pendant l'entraînement. Le détail est dans les logs du backend."
NON_CONFORMING_DETAIL = (
    "Erreur interne : le résultat de l'entraînement ne respecte pas le contrat de POST /train. "
    "Le détail est dans les logs du backend."
)

TrainAndPromote = Callable[[str], Any]

FLOAT_FIELDS = ("metric_value", "promotion_threshold", "decision_threshold")

_LOCKS_GUARD = threading.Lock()
_MISSING = object()

router = APIRouter(tags=["entraînement"])


def _log_base_exception(origin: str, error: BaseException) -> None:
    """Journalise une BaseException du code de R2 : son type et ses cadres dans src/, jamais son message."""
    logger.error("%s a levé %s : entraînement interrompu.\n%s", origin, type(error).__name__, _app_frames(error))


def get_train_and_promote() -> TrainAndPromote:
    """Adaptateur vers train_and_promote(preset) du lot P3, importée seulement à la demande.

    L'import paresseux laisse l'API démarrer sans le module (et sans MLflow chargé) : tant que
    R2 n'est pas livrée, la route répond 501. Une erreur d'import interne au module (une
    dépendance qui lui manque) n'est pas masquée : elle donne une 500 et ses cadres dans les
    logs. Les tests remplacent cet adaptateur par app.dependency_overrides.
    """
    try:
        module = importlib.import_module(PROMOTION_MODULE)
    except ModuleNotFoundError as error:
        if error.name not in {PROMOTION_MODULE, "src.registry"}:
            # Le nom d'un module n'est pas une donnée de la demande : il peut être journalisé.
            logger.error("Import de %s impossible : le module %s est introuvable.", PROMOTION_MODULE, error.name)
            raise
        logger.warning("POST /train (501) : %s n'existe pas encore (tâche R2).", PROMOTION_MODULE)
        raise ApiError(501, ErrorCode.NOT_IMPLEMENTED, NOT_DELIVERED_DETAIL) from None
    except Exception:
        raise  # le middleware de A1 en fait une 500 au format commun
    except BaseException as error:  # noqa: BLE001 - sys.exit() pendant l'import du module, par exemple
        _log_base_exception(f"L'import de {PROMOTION_MODULE}", error)
        raise ApiError(500, ErrorCode.INTERNAL_ERROR, FAILED_DETAIL) from None
    function = getattr(module, PROMOTION_FUNCTION, None)
    if not callable(function):
        logger.warning("POST /train (501) : %s ne définit pas %s (tâche R2).", PROMOTION_MODULE, PROMOTION_FUNCTION)
        raise ApiError(501, ErrorCode.NOT_IMPLEMENTED, NOT_DELIVERED_DETAIL)
    return function


def training_lock(state: State) -> threading.Lock:
    """Verrou d'entraînement de l'application, créé au premier appel, sans risque de doublon."""
    with _LOCKS_GUARD:
        lock = getattr(state, "training_lock", None)
        if lock is None:
            lock = threading.Lock()
            state.training_lock = lock
        return lock


def to_response(result: Any) -> TrainResponse:
    """Résultat de train_and_promote converti en TrainResponse, ou ApiError 500 s'il n'est pas conforme.

    Le résultat peut être une TrainResponse, un dictionnaire ou un objet ayant les attributs
    du contrat. Seuls les champs du contrat sont lus : une information en plus est ignorée.
    """
    if inspect.isawaitable(result):
        if inspect.iscoroutine(result):
            result.close()
        logger.error("%s a renvoyé un objet asynchrone : elle doit être synchrone.", PROMOTION_FUNCTION)
        raise ApiError(500, ErrorCode.INTERNAL_ERROR, NON_CONFORMING_DETAIL)
    if isinstance(result, Mapping):
        values = {key: result.get(key, _MISSING) for key in TrainResponse.model_fields}
    else:
        values = {key: getattr(result, key, _MISSING) for key in TrainResponse.model_fields}
    candidate = {key: value for key, value in values.items() if value is not _MISSING}
    try:
        response = TrainResponse.model_validate(candidate)
    except ValidationError as error:
        # Seuls les noms des champs du contrat sont journalisés, jamais les valeurs.
        fields = sorted({".".join(map(str, item["loc"])) or "(ensemble)" for item in error.errors()})
        logger.error(
            "%s a renvoyé un résultat non conforme à TrainResponse (%s) : %s.",
            PROMOTION_FUNCTION,
            type(result).__name__,
            ", ".join(fields),
        )
        raise ApiError(500, ErrorCode.INTERNAL_ERROR, NON_CONFORMING_DETAIL) from None
    # Le contrat accepte NaN et l'infini, que le JSON écrirait null : ils sont refusés ici.
    not_finite = [key for key in FLOAT_FIELDS if not math.isfinite(getattr(response, key))]
    if not_finite:
        logger.error(
            "%s a renvoyé un résultat non conforme à TrainResponse (%s) : %s non fini.",
            PROMOTION_FUNCTION,
            type(result).__name__,
            ", ".join(not_finite),
        )
        raise ApiError(500, ErrorCode.INTERNAL_ERROR, NON_CONFORMING_DETAIL)
    return response


def reload_served_model(state: State) -> bool:
    """Appelle app.state.reload_model() (A2). Renvoie True si le rechargement a réussi ; ne lève jamais d'Exception."""
    reload = getattr(state, "reload_model", None)
    if reload is None:
        logger.warning(
            "app.state.reload_model n'est pas défini (tâche A2) : /predict garde le modèle chargé au démarrage."
        )
        return False
    if not callable(reload):
        logger.warning("app.state.reload_model n'est pas une fonction : /predict garde le modèle précédent.")
        return False
    try:
        outcome = reload()
    except BaseException as error:  # noqa: BLE001 - l'entraînement a réussi : la réponse reste 200
        logger.error(
            "Rechargement du modèle servi en échec (%s) : /predict garde le modèle précédent.\n%s",
            type(error).__name__,
            _app_frames(error),
        )
        return False
    if inspect.isawaitable(outcome):
        if inspect.iscoroutine(outcome):
            outcome.close()
        logger.warning("app.state.reload_model doit être synchrone : le modèle servi n'a pas été rechargé.")
        return False
    logger.info("Modèle servi rechargé après l'entraînement.")
    return True


@router.post(
    "/train",
    response_model=TrainResponse,
    dependencies=[Depends(require_api_key)],
    responses={
        401: {"model": ErrorResponse, "description": "En-tête X-API-Key absent."},
        403: {"model": ErrorResponse, "description": "Clé fausse, ou clé non configurée sur le serveur."},
        409: {"model": ErrorResponse, "description": "Un entraînement est déjà en cours."},
        422: {"model": ErrorResponse, "description": "Préréglage inconnu."},
        500: {"model": ErrorResponse, "description": "Entraînement en échec ou résultat non conforme."},
        501: {"model": ErrorResponse, "description": "train_and_promote (tâche R2) pas encore livrée."},
    },
    summary="Entraîner un modèle et le promouvoir s'il atteint le seuil",
)
def train(
    request: Request,
    train_and_promote: Annotated[TrainAndPromote, Depends(get_train_and_promote)],
    payload: TrainRequest | None = None,
) -> TrainResponse:
    """Entraîne avec le préréglage demandé, puis promeut le modèle s'il atteint le seuil.

    Le corps est facultatif : sans corps, le préréglage est default. La réponse arrive à la
    fin de l'entraînement ; un second appel pendant ce temps reçoit 409.
    """
    preset = (payload or TrainRequest()).preset
    state = request.app.state
    lock = training_lock(state)
    if not lock.acquire(blocking=False):
        logger.info("POST /train refusé (409) : un entraînement est déjà en cours.")
        raise ApiError(409, ErrorCode.TRAINING_IN_PROGRESS, IN_PROGRESS_DETAIL)
    started = time.perf_counter()
    # L'alias @champion a pu bouger si l'entraînement échoue ou renvoie un résultat non
    # conforme : le modèle servi est alors rechargé aussi, par prudence.
    may_have_promoted = True
    try:
        logger.info("Entraînement lancé (préréglage %s).", preset)
        try:
            result = train_and_promote(preset)
        except Exception:
            raise  # le middleware de A1 en fait une 500 au format commun
        except BaseException as error:  # noqa: BLE001 - traitée puis convertie en ApiError
            # sys.exit() dans le code d'entraînement ne doit pas sortir du format d'erreur commun.
            _log_base_exception(PROMOTION_FUNCTION, error)
            raise ApiError(500, ErrorCode.INTERNAL_ERROR, FAILED_DETAIL) from None
        response = to_response(result)
        may_have_promoted = response.promoted
        elapsed = round(time.perf_counter() - started, 1)
        if response.promoted:
            logger.info(
                "Entraînement terminé en %s s (préréglage %s, run %r) : %r %.4f (seuil %.4f), version %s promue @champion.",
                elapsed,
                preset,
                response.run_id,
                response.metric_name,
                response.metric_value,
                response.promotion_threshold,
                response.model_version,
            )
        else:
            logger.info(
                "Entraînement terminé en %s s (préréglage %s, run %r) : %r %.4f (seuil %.4f), modèle refusé, @champion inchangé.",
                elapsed,
                preset,
                response.run_id,
                response.metric_name,
                response.metric_value,
                response.promotion_threshold,
            )
        return response
    finally:
        try:
            if may_have_promoted:
                reload_served_model(state)
        finally:
            lock.release()

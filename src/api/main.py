"""Application FastAPI de Neuraxion (tâche A1).

Lancement : uv run uvicorn src.api.main:app

Au démarrage, l'application lit le contrat de données (contracts/features.yaml) et les
exemples (contracts/examples.json), construit la classe de requête de POST /predict avec
build_request_model(), puis inclut les routeurs du contrat C4. Un contrat absent ou invalide
empêche le démarrage, avec un message en une ligne qui nomme le fichier.

Raccord avec A2 : le modèle servi est décrit par app.state.model_info (ModelInfo ou None).
A2 le renseigne depuis le routeur predict (paramètre lifespan de l'APIRouter, que FastAPI
fusionne avec celui de l'application) ; aucune autre tâche ne modifie ce fichier.
"""

import asyncio
import inspect
import json
import logging
import threading
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel, ValidationError

from src.api import feedback, predict, train
from src.api.errors import install_error_handlers
from src.api.schemas import HealthResponse, ModelInfo, SchemaResponse, build_request_model
from src.db.connection import database_is_up

logger = logging.getLogger("neuraxion.api")

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "contracts" / "features.yaml"
EXAMPLES_PATH = ROOT / "contracts" / "examples.json"
UNSPECIFIED_VERSION = "non précisée"
HEALTH_CHECK_TIMEOUT_SECONDS = 2.0


class ContractError(RuntimeError):
    """Le contrat de données ou ses exemples empêchent le démarrage de l'API."""


def configure_logging() -> None:
    """Logs lisibles sur la sortie d'erreur, et coupure du journal d'accès d'uvicorn.

    Le journal d'accès d'uvicorn écrit l'adresse IP du client et l'URL complète : il est
    remplacé par la ligne de errors.py (méthode, route, statut, durée). La coupure est
    refaite au démarrage (lifespan), car une configuration des logs postérieure à la
    création de l'application, comme celle de uvicorn.run(app), la réactiverait.
    """
    app_logger = logging.getLogger("neuraxion")
    if not app_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s : %(message)s"))
        app_logger.addHandler(handler)
    app_logger.setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").disabled = True


def _read_text(path: Path, what: str, plural: bool = False) -> str:
    unreadable = "illisibles" if plural else "illisible"
    try:
        if not path.is_file():
            missing = "introuvables" if plural else "introuvable"
            raise ContractError(f"{what} {missing} : {path}. Ce fichier est livré par le lot P1 (tâche C1).")
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ContractError(f"{what} {unreadable} ({path}) : le fichier n'est pas encodé en UTF-8.") from error
    except OSError as error:
        raise ContractError(f"{what} {unreadable} ({path}) : accès impossible ({type(error).__name__}).") from error


def load_contract(path: Path) -> dict[str, Any]:
    """Lit le contrat YAML. Lève ContractError si le fichier manque ou n'est pas un dictionnaire."""
    text = _read_text(path, "Contrat de données")
    try:
        contract = yaml.safe_load(text)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        where = f" (ligne {mark.line + 1}, colonne {mark.column + 1})" if mark is not None else ""
        raise ContractError(f"Contrat de données illisible ({path}) : syntaxe YAML invalide{where}.") from error
    if not isinstance(contract, dict):
        raise ContractError(f"Contrat de données invalide ({path}) : un dictionnaire YAML est attendu.")
    version = contract.get("version")
    if version is not None and (isinstance(version, bool) or not isinstance(version, str | int)):
        raise ContractError(
            f"Contrat de données invalide ({path}) : version doit être un texte, écrit entre guillemets "
            '(version: "1.10" et non 1.10).'
        )
    return contract


def load_examples(path: Path, request_model: type[BaseModel]) -> dict[str, dict[str, Any]]:
    """Lit les demandes d'exemple (profil -> demande) et vérifie chacune avec le modèle de requête."""
    text = _read_text(path, "Exemples", plural=True)
    try:
        examples = json.loads(text)
    except json.JSONDecodeError as error:
        raise ContractError(
            f"Exemples illisibles ({path}) : JSON invalide (ligne {error.lineno}, colonne {error.colno})."
        ) from error
    if not isinstance(examples, dict) or not examples:
        raise ContractError(f"Exemples invalides ({path}) : un objet JSON profil -> demande est attendu.")
    for profile, example in examples.items():
        if not isinstance(example, dict):
            raise ContractError(f"Exemple « {profile} » invalide ({path}) : un objet JSON est attendu.")
        try:
            request_model.model_validate(example)
        except ValidationError as error:
            fields = ", ".join(sorted({".".join(map(str, item["loc"])) or "?" for item in error.errors()}))
            raise ContractError(
                f"Exemple « {profile} » non conforme au contrat ({path}) : champs {fields}."
            ) from error
    return examples


def _build_request_model(contract: dict[str, Any], path: Path) -> tuple[type[BaseModel], dict[str, Any]]:
    """Classe de requête et son JSON Schema. Toute erreur devient une ContractError."""
    try:
        model = build_request_model(contract)
        return model, model.model_json_schema()
    except ValueError as error:
        raise ContractError(f"Contrat de données invalide ({path}) : {error}") from error
    except Exception as error:  # noqa: BLE001 - filet : Pydantic peut refuser un cas non prévu
        raise ContractError(f"Contrat de données invalide ({path}) : {type(error).__name__}.") from error


def _served_model(state: Any) -> ModelInfo | None:
    """Modèle servi tel que A2 l'a renseigné, validé ; None s'il est absent ou mal formé."""
    model = state.model_info
    if model is None:
        return None
    try:
        # Seuls les trois attributs attendus sont lus et validés, y compris pour une instance
        # de ModelInfo construite sans validation (model_construct).
        read = model.get if isinstance(model, dict) else lambda key: getattr(model, key, None)
        candidate = {key: read(key) for key in ModelInfo.model_fields}
        return ModelInfo.model_validate(candidate)
    except ValidationError:
        if not state.model_info_warned:
            logger.warning("app.state.model_info est mal formé : /health le traite comme absent.")
            state.model_info_warned = True
        return None


def _run_check(check: Callable[[], Any], future: Future) -> None:
    """Exécute le contrôle dans le fil démon. Le Future reçoit le résultat, jamais d'exception."""
    try:
        result = check()
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            logger.debug("database_check a renvoyé un objet asynchrone : base comptée indisponible.")
            result = False
    except BaseException as error:  # noqa: BLE001 - y compris CancelledError levée par le contrôle
        logger.debug("Contrôle de la base en échec (%s).", type(error).__name__)
        result = False
    future.set_result(result)


def _current_check(state: Any) -> Future:
    """Contrôle de la base en cours, ou un nouveau s'il n'y en a pas.

    Un seul contrôle tourne à la fois, dans un fil démon : un contrôle bloqué n'accumule pas
    de travaux en attente et n'empêche pas le processus de s'arrêter. Les appels de /health
    qui arrivent pendant ce temps attendent le même contrôle, au plus le délai.
    """
    with state.database_lock:
        future = state.database_future
        if future is None or future.done():
            future = Future()
            future.set_running_or_notify_cancel()
            state.database_future = future
            threading.Thread(
                target=_run_check, args=(state.database_check, future), daemon=True, name="health"
            ).start()
    return future


def _record_database_state(state: Any, database_up: bool) -> None:
    """Journalise un changement d'état de la base, une seule fois, même avec des appels simultanés."""
    with state.database_lock:
        if database_up == state.database_up:
            return
        state.database_up = database_up
    if database_up:
        logger.info("PostgreSQL répond.")
    else:
        logger.warning("PostgreSQL ne répond pas ou n'est pas configuré (variables POSTGRES_*).")


def create_app(
    contract_path: Path | str | None = None,
    examples_path: Path | str | None = None,
    database_check: Callable[[], bool] | None = None,
) -> FastAPI:
    """Construit l'application.

    Paramètres
        contract_path : contrat de données, contracts/features.yaml par défaut.
        examples_path : demandes d'exemple, contracts/examples.json par défaut.
        database_check : fonction synchrone sans argument, qui renvoie True si la base
            répond ; par défaut database_is_up(), qui lit les variables POSTGRES_*.

    Lève
        ContractError si le contrat ou les exemples sont absents ou invalides.
        TypeError si database_check est une fonction asynchrone.
    """
    if database_check is not None and (
        inspect.iscoroutinefunction(database_check)
        or inspect.iscoroutinefunction(getattr(database_check, "__call__", None))
    ):
        raise TypeError("database_check doit être une fonction synchrone.")
    configure_logging()
    contract_path = Path(contract_path) if contract_path is not None else CONTRACT_PATH
    examples_path = Path(examples_path) if examples_path is not None else EXAMPLES_PATH

    contract = load_contract(contract_path)
    request_model, request_schema = _build_request_model(contract, contract_path)
    examples = load_examples(examples_path, request_model)
    version = contract.get("version")
    schema = SchemaResponse(
        contract_version=UNSPECIFIED_VERSION if version is None else str(version),
        request_schema=request_schema,
        examples=examples,
    )

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        logging.getLogger("uvicorn.access").disabled = True
        yield

    app = FastAPI(
        title="Neuraxion",
        summary="Probabilité de défaut d'un prêt et décision d'octroi.",
        description="Contrat détaillé dans docs/api.md. Le formulaire se construit avec GET /schema.",
        lifespan=lifespan,
    )
    app.state.model_info = None
    app.state.model_info_warned = False
    app.state.schema = schema
    app.state.database_check = database_check or database_is_up
    app.state.database_up = None
    app.state.database_future = None
    app.state.database_lock = threading.Lock()

    install_error_handlers(app)
    app.include_router(predict.build_router(request_model))
    app.include_router(train.router)
    app.include_router(feedback.router)

    @app.get("/health", response_model=HealthResponse, tags=["état"])
    async def health(request: Request) -> HealthResponse:
        """État de l'API, de la base et du modèle servi. Répond toujours 200, en 2 s au plus."""
        state = request.app.state
        try:
            future = _current_check(state)
            result = await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(future)), timeout=HEALTH_CHECK_TIMEOUT_SECONDS
            )
            database_up = result is True
        except TimeoutError:
            logger.debug("Contrôle de la base trop long (plus de %s s).", HEALTH_CHECK_TIMEOUT_SECONDS)
            database_up = False
        _record_database_state(state, database_up)
        model = _served_model(state)
        return HealthResponse(
            status="ok" if database_up and model is not None else "degraded",
            database="ok" if database_up else "unavailable",
            model=model,
        )

    # HEAD /health pour les contrôles de santé écrits avec curl -I ou wget --spider.
    app.add_api_route("/health", health, methods=["HEAD"], include_in_schema=False)

    @app.get("/schema", response_model=SchemaResponse, tags=["contrat"])
    def get_schema(request: Request) -> SchemaResponse:
        """JSON Schema de la requête de POST /predict et demandes d'exemple pour le formulaire."""
        return request.app.state.schema

    logger.info(
        "API prête : contrat %s, %d variables, %d exemples.",
        schema.contract_version,
        len(contract["features"]),
        len(examples),
    )
    return app


def __getattr__(name: str) -> FastAPI:
    # L'application n'est construite qu'au premier accès à src.api.main.app (par uvicorn) :
    # importer create_app dans les tests ne lit donc pas contracts/.
    if name == "app":
        try:
            application = create_app()
        except ContractError as error:
            # Une ligne claire plutôt qu'une pile d'appels de plusieurs dizaines de lignes.
            logger.critical("Démarrage impossible : %s", error)
            raise SystemExit(1) from None
        globals()["app"] = application
        return application
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

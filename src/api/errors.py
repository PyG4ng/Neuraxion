"""Erreurs de l'API au format commun (ErrorResponse) et journal des requêtes.

Les routes signalent une erreur maîtrisée en levant ApiError, par exemple :

    raise ApiError(503, ErrorCode.MODEL_UNAVAILABLE, "Aucun modèle validé n'est disponible. ...")

Ni la réponse ni les logs ne recopient les données de la demande : les messages de
validation ne citent que le nom du champ, et les logs ne citent que les champs du contrat,
le modèle de route et une méthode HTTP connue.
"""

import json
import logging
import time
import traceback
import typing
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.api.schemas import ErrorCode, ErrorResponse, FieldError

logger = logging.getLogger("neuraxion.api")

SOURCE_ROOT = Path(__file__).resolve().parents[1]  # dossier src/
LOCATION_ROOTS = ("body", "query", "path", "header", "cookie")
BODY_FIELD = "corps"
UNKNOWN_ROUTE = "<route inconnue>"
UNKNOWN_METHOD = "<méthode inconnue>"
KNOWN_METHODS = frozenset({"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"})
MAX_BODY_BYTES = 64 * 1024  # une demande de crédit tient en quelques centaines d'octets
MAX_FIELD_ERRORS = 50

STATUS_CODES = {
    400: ErrorCode.BAD_REQUEST,
    401: ErrorCode.UNAUTHORIZED,
    403: ErrorCode.FORBIDDEN,
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    422: ErrorCode.VALIDATION_ERROR,
    501: ErrorCode.NOT_IMPLEMENTED,
}

STATUS_MESSAGES = {
    400: "Requête mal formée.",
    401: "Authentification requise.",
    403: "Accès refusé.",
    404: "Route inconnue. La liste des routes est dans /docs.",
    405: "Méthode non autorisée pour cette route.",
}

TRANSLATIONS = {
    "extra_forbidden": "champ inconnu",
    "missing": "champ obligatoire manquant",
    "float_type": "doit être un nombre",
    "float_parsing": "doit être un nombre",
    "finite_number": "doit être un nombre fini",
    "int_type": "doit être un nombre entier",
    "int_parsing": "doit être un nombre entier",
    "int_from_float": "doit être un nombre entier",
    "string_type": "doit être une chaîne de caractères",
    "string_unicode": "doit être une chaîne de caractères valide",
    "uuid_type": "doit être un identifiant UUID",
    "uuid_parsing": "doit être un identifiant UUID",
    "json_invalid": "JSON invalide",
    "model_attributes_type": "le corps doit être un objet JSON, envoyé avec l'en-tête Content-Type: application/json",
    "dict_type": "doit être un objet JSON",
    "model_type": "doit être un objet JSON",
}
BOUNDS = {
    "greater_than_equal": ("supérieur ou égal à", "ge"),
    "greater_than": ("strictement supérieur à", "gt"),
    "less_than_equal": ("inférieur ou égal à", "le"),
    "less_than": ("strictement inférieur à", "lt"),
}


class ApiError(Exception):
    """Erreur maîtrisée, renvoyée telle quelle au format commun."""

    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        detail: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.headers = dict(headers) if headers else None


def error_response(
    status_code: int,
    code: ErrorCode,
    detail: str,
    errors: list[FieldError] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Réponse JSON au format commun."""
    body = ErrorResponse(detail=detail, code=code, errors=errors)
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"), headers=headers)


def field_name(location: Iterable[Any]) -> str:
    """Nom lisible d'un champ à partir de la localisation Pydantic, sans la racine body."""
    parts = list(location)
    if parts and parts[0] in LOCATION_ROOTS:
        parts = parts[1:]
    if not parts:
        return BODY_FIELD
    return ".".join(str(part) for part in parts)


def _field_of(error: Mapping[str, Any]) -> str:
    # Erreur de décodage JSON : la localisation est une position dans le texte, pas un champ.
    if error.get("type") == "json_invalid":
        return BODY_FIELD
    return field_name(error.get("loc", ()))


def _number(value: Any) -> str:
    """Borne lisible : 1000 plutôt que 1000.0."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def french_message(error: Mapping[str, Any]) -> str:
    """Message français d'une erreur Pydantic, sans la valeur reçue."""
    kind = error.get("type", "")
    context = error.get("ctx") or {}
    if kind in BOUNDS:
        words, key = BOUNDS[kind]
        return f"doit être {words} {_number(context.get(key))}"
    if kind == "literal_error":
        expected = str(context.get("expected", "")).replace(" or ", " ou ")
        return f"valeur non permise, valeurs possibles : {expected}"
    if kind == "not_allowed":
        # Erreur de schemas._member_of : elle ne cite que les valeurs permises par le contrat.
        return f"valeur non permise, valeurs possibles : {context.get('allowed', '')}"
    # value_error et tout autre type : un validateur peut citer la valeur reçue.
    return TRANSLATIONS.get(kind, "valeur invalide")


def _declared_fields(scope: Scope) -> frozenset[str]:
    """Noms des champs déclarés par le modèle du corps de la route (le contrat pour /predict)."""
    body_field = getattr(scope.get("route"), "body_field", None)
    annotation = getattr(getattr(body_field, "field_info", None), "annotation", None)
    candidates = [annotation, *typing.get_args(annotation)]  # TrainRequest | None, par exemple
    for candidate in candidates:
        fields = getattr(candidate, "model_fields", None)
        if isinstance(fields, dict):
            return frozenset(fields)
    return frozenset()


def _body_location(error: Mapping[str, Any]) -> list[Any]:
    """Localisation dans le corps, sans la seule racine (body, query...)."""
    location = list(error.get("loc", ()))
    if location and location[0] in LOCATION_ROOTS:
        location = location[1:]
    return location


def _is_body_level(error: Mapping[str, Any]) -> bool:
    """Erreur qui porte sur le corps entier (JSON mal formé, corps absent ou non objet)."""
    return error.get("type") == "json_invalid" or not _body_location(error)


DOCUMENTATION_PATHS = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"})


def _route_path(scope: Scope) -> str:
    """Modèle de chemin de la route (/predict), jamais le chemin brut envoyé par le client.

    Les pages de documentation de FastAPI ne renseignent pas scope["route"] : leur chemin, fixe,
    est repris s'il fait partie de la liste connue. Une requête refusée avant le routage (413
    annoncée dans l'en-tête) reste <route inconnue>.
    """
    route = scope.get("route")
    if route is not None:
        return getattr(route, "path", UNKNOWN_ROUTE)
    if "endpoint" in scope and scope.get("path") in DOCUMENTATION_PATHS:
        return scope["path"]
    return UNKNOWN_ROUTE


def _method(scope: Scope) -> str:
    """Méthode HTTP connue ; une méthode inventée par le client n'est pas recopiée."""
    method = scope.get("method", "")
    return method if method in KNOWN_METHODS else UNKNOWN_METHOD


def _app_frames(error: BaseException) -> str:
    """Cadres de pile situés dans src/ : fichier, ligne, fonction. Ni message ni variables."""
    frames = []
    for frame in traceback.extract_tb(error.__traceback__):
        path = Path(frame.filename).resolve()
        if SOURCE_ROOT in path.parents:
            frames.append(f"  {path.relative_to(SOURCE_ROOT.parent).as_posix()}:{frame.lineno} dans {frame.name}")
    return "\n".join(frames) if frames else "  (aucun cadre dans src/)"


async def _api_error_handler(request: Request, error: ApiError) -> JSONResponse:
    return error_response(error.status_code, error.code, error.detail, headers=error.headers)


async def _validation_handler(request: Request, error: RequestValidationError) -> JSONResponse:
    raw_errors = list(error.errors())
    count = len(raw_errors)
    field_errors = [
        FieldError(field=_field_of(item), message=french_message(item)) for item in raw_errors[:MAX_FIELD_ERRORS]
    ]
    if all(_is_body_level(item) for item in raw_errors):
        detail = "La demande a été refusée : le corps de la requête n'est pas un objet JSON valide."
    else:
        detail = "La demande a été refusée : " + (
            "un champ ne respecte pas le contrat." if count == 1 else f"{count} champs ne respectent pas le contrat."
        )
    if count > MAX_FIELD_ERRORS:
        detail += f" Seules les {MAX_FIELD_ERRORS} premières erreurs sont listées."
    detail += " Les champs attendus sont décrits par GET /schema."
    # Dans les logs, seuls les champs déclarés par la route sont nommés (premier niveau) : un
    # champ inconnu ou une clé de dictionnaire est un nom choisi par le client, donc une donnée.
    declared = _declared_fields(request.scope)
    known: set[str] = set()
    unknown = 0
    for item in raw_errors:
        if _is_body_level(item):
            known.add(f"{BODY_FIELD} ({item.get('type')})")
            continue
        top = _body_location(item)[0]
        if item.get("type") != "extra_forbidden" and top in declared:
            # Seul le nom de premier niveau est écrit : une clé de dictionnaire plus bas
            # serait un nom choisi par le client.
            known.add(f"{top} ({item.get('type')})")
        else:
            unknown += 1
    known = sorted(known)
    logger.info(
        "Demande refusée (422) sur %s %s : %s ; autres champs (inconnus ou non déclarés) : %d.",
        _method(request.scope),
        _route_path(request.scope),
        ", ".join(known[:MAX_FIELD_ERRORS]) if known else "aucun champ du contrat",
        unknown,
    )
    return error_response(422, ErrorCode.VALIDATION_ERROR, detail, errors=field_errors)


async def _http_error_handler(request: Request, error: StarletteHTTPException) -> Response:
    status = error.status_code
    headers = getattr(error, "headers", None)
    if status < 400:
        return Response(status_code=status, headers=headers)
    code = STATUS_CODES.get(status, ErrorCode.INTERNAL_ERROR if status >= 500 else ErrorCode.BAD_REQUEST)
    detail = STATUS_MESSAGES.get(status)
    if detail is None:
        detail = error.detail if isinstance(error.detail, str) else "Requête refusée."
    return error_response(status, code, detail, headers=headers)


class _BodyTooLarge(Exception):
    """Interne : le corps dépasse la taille maximale."""


class RequestGuard:
    """Middleware ASGI : journal des requêtes, taille maximale du corps, erreurs imprévues.

    Middleware ASGI pur, et non @app.middleware("http") : une exception levée après le début
    de la réponse (tâche d'arrière-plan, réponse en flux) est aussi rattrapée, et seul son
    type est journalisé.
    """

    def __init__(self, app: ASGIApp, max_body_bytes: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        state: dict[str, Any] = {"status": None, "started": False, "too_large": False}
        received = 0

        async def send_and_record(message: Message) -> None:
            if state["too_large"]:
                # FastAPI transforme l'échec de lecture du corps en 400 : cette réponse est
                # écartée et remplacée par la 413 ci-dessous.
                return
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                state["started"] = True
            await send(message)

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    state["too_large"] = True
                    raise _BodyTooLarge
            return message

        try:
            if self._declared_length(scope) > self.max_body_bytes:
                state["too_large"] = True
                raise _BodyTooLarge
            await self.app(scope, limited_receive, send_and_record)
            if state["too_large"]:
                raise _BodyTooLarge
        except _BodyTooLarge:
            state["too_large"] = False
            if state["started"]:
                logger.warning(
                    "Réponse interrompue sur %s %s : corps de plus de %d Ko lu après le début de la réponse.",
                    _method(scope),
                    _route_path(scope),
                    self.max_body_bytes // 1024,
                )
            else:
                await self._send_too_large(send_and_record)
        except Exception as error:  # noqa: BLE001 - toute erreur imprévue devient une 500 propre
            if state["too_large"]:
                # L'aval a échoué en lisant un corps trop gros : la cause est la taille.
                state["too_large"] = False
                if not state["started"]:
                    await self._send_too_large(send_and_record)
                elapsed_ms = round((time.perf_counter() - started) * 1000)
                logger.info("%s %s -> %s (%d ms)", _method(scope), _route_path(scope), state["status"], elapsed_ms)
                return
            # Le message de l'exception peut recopier la demande : seuls son type et les
            # cadres de src/ (fichier, ligne, fonction) sont journalisés.
            logger.error(
                "Erreur interne sur %s %s : %s\n%s",
                _method(scope),
                _route_path(scope),
                type(error).__name__,
                _app_frames(error),
            )
            if not state["started"]:
                await self._send_error(
                    send_and_record,
                    500,
                    ErrorCode.INTERNAL_ERROR,
                    "Erreur interne. Le détail est dans les logs du backend.",
                )
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        logger.info("%s %s -> %s (%d ms)", _method(scope), _route_path(scope), state["status"], elapsed_ms)

    async def _send_too_large(self, send: Send) -> None:
        await self._send_error(
            send,
            413,
            ErrorCode.PAYLOAD_TOO_LARGE,
            f"Corps de la requête trop volumineux (plus de {self.max_body_bytes // 1024} Ko).",
        )

    @staticmethod
    def _declared_length(scope: Scope) -> int:
        for key, value in scope.get("headers", []):
            if key == b"content-length":
                try:
                    return int(value)
                except ValueError:
                    return 0
        return 0

    @staticmethod
    async def _send_error(send: Send, status: int, code: ErrorCode, detail: str) -> None:
        body = json.dumps(ErrorResponse(detail=detail, code=code).model_dump(mode="json")).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})


def install_error_handlers(app: FastAPI) -> None:
    """Branche les gestionnaires d'erreurs et le middleware des requêtes sur l'application."""
    app.add_exception_handler(ApiError, _api_error_handler)
    app.add_exception_handler(RequestValidationError, _validation_handler)
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)
    app.add_middleware(RequestGuard)

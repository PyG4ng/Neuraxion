"""Clé d'API de POST /train (tâche A3).

La clé attendue est lue dans la variable d'environnement TRAIN_API_KEY (voir .env.example) à
chaque requête. Le client l'envoie dans l'en-tête X-API-Key :

- en-tête absent ou vide : 401 UNAUTHORIZED ;
- clé fausse, ou TRAIN_API_KEY absente ou vide sur le serveur : 403 FORBIDDEN, avec le même
  message dans les deux cas (la route reste fermée tant que la clé n'est pas configurée).

La comparaison se fait à temps constant, sur les empreintes SHA-256 des deux clés : sa durée
ne dépend ni du nombre de caractères justes ni de la longueur de la clé attendue (seul le
calcul de l'empreinte de la clé reçue dépend de sa longueur, ce qui ne révèle rien). Ni la clé
attendue ni la clé reçue n'apparaissent dans les réponses ou dans les logs.
"""

import hashlib
import hmac
import logging
import os
from typing import Annotated

from fastapi import Request, Security
from fastapi.security import APIKeyHeader

from src.api.errors import ApiError
from src.api.schemas import ErrorCode

logger = logging.getLogger("neuraxion.api")

API_KEY_HEADER = "X-API-Key"
API_KEY_ENV = "TRAIN_API_KEY"

MISSING_KEY_DETAIL = f"Authentification requise : envoyez la clé d'entraînement dans l'en-tête {API_KEY_HEADER}."
REFUSED_KEY_DETAIL = (
    f"Clé {API_KEY_HEADER} refusée : elle est fausse, ou {API_KEY_ENV} n'est pas configurée sur le serveur "
    "(voir les logs du backend)."
)

# auto_error=False : l'absence d'en-tête est signalée au format commun, pas par FastAPI.
# Le schéma de sécurité est publié dans /openapi.json (bouton Authorize de /docs).
api_key_header = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    scheme_name="TrainApiKey",
    description=f"Clé d'entraînement, comparée à la variable {API_KEY_ENV} du serveur.",
)


def configured_key() -> str | None:
    """Clé attendue, sans les espaces autour ; None si TRAIN_API_KEY est absente ou vide."""
    key = os.environ.get(API_KEY_ENV, "").strip()
    return key or None


def _digest(raw: bytes) -> bytes:
    return hashlib.sha256(raw).digest()


def keys_match(received: str, expected: str) -> bool:
    """Compare deux clés à temps constant.

    L'en-tête reçu est décodé en latin-1 par le serveur : il est réencodé en latin-1 pour
    retrouver les octets envoyés, et la clé attendue est encodée en UTF-8. Une clé non ASCII
    écrite en UTF-8 des deux côtés est donc reconnue. En appel direct, received doit donc être
    la chaîne telle que le serveur la décode, pas le texte d'origine.
    """
    try:
        received_bytes = received.encode("latin-1")
    except UnicodeEncodeError:
        received_bytes = received.encode("utf-8", errors="surrogatepass")
    expected_bytes = expected.encode("utf-8", errors="surrogatepass")
    return hmac.compare_digest(_digest(received_bytes), _digest(expected_bytes))


def require_api_key(request: Request, received: Annotated[str | None, Security(api_key_header)]) -> None:
    """Dépendance de POST /train : laisse passer la requête si la clé est la bonne, sinon lève ApiError.

    Plusieurs en-têtes X-API-Key sont refusés (403) : seule la première valeur serait lue, si
    bien qu'un intermédiaire qui ajoute le sien pourrait changer la clé retenue.
    """
    # Contrôlé en premier : deux en-têtes dont le premier est vide sont aussi refusés en 403.
    if len(request.headers.getlist(API_KEY_HEADER)) > 1:
        logger.warning("POST /train refusé (403) : plusieurs en-têtes %s.", API_KEY_HEADER)
        raise ApiError(403, ErrorCode.FORBIDDEN, REFUSED_KEY_DETAIL)
    if not received:
        logger.info("POST /train refusé (401) : en-tête %s absent.", API_KEY_HEADER)
        raise ApiError(401, ErrorCode.UNAUTHORIZED, MISSING_KEY_DETAIL, headers={"WWW-Authenticate": "APIKey"})
    expected = configured_key()
    if expected is None:
        # La comparaison est faite quand même, pour que la durée de la réponse ne révèle pas ce cas.
        keys_match(received, received)
        logger.warning(
            "POST /train refusé (403) : %s n'est pas configurée sur le serveur, la route reste fermée.", API_KEY_ENV
        )
        raise ApiError(403, ErrorCode.FORBIDDEN, REFUSED_KEY_DETAIL)
    if not keys_match(received, expected):
        logger.warning("POST /train refusé (403) : clé %s fausse.", API_KEY_HEADER)
        raise ApiError(403, ErrorCode.FORBIDDEN, REFUSED_KEY_DETAIL)

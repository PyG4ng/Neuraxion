"""Connexion à PostgreSQL à partir des variables d'environnement (.env.example).

Le mot de passe n'est jamais journalisé ni renvoyé dans un message d'erreur.
"""

import logging
import os
from collections.abc import Mapping

import psycopg

logger = logging.getLogger("neuraxion.db")

REQUIRED_VARIABLES = ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB")
CONNECT_TIMEOUT_SECONDS = 2


class DatabaseNotConfigured(RuntimeError):
    """Une variable obligatoire de connexion manque."""


def connection_params(environ: Mapping[str, str] | None = None) -> dict[str, str | int]:
    """Paramètres de psycopg.connect lus dans l'environnement.

    POSTGRES_HOST vaut localhost et POSTGRES_PORT 5432 par défaut. Dans Compose, ce sont
    postgres et 5432 ; depuis la machine hôte, 127.0.0.1 et 5433 (sous Windows, localhost essaie
    d'abord ::1, où Compose ne publie pas le port, et chaque connexion perd 2 s).

    Lève
        DatabaseNotConfigured si POSTGRES_USER, POSTGRES_PASSWORD ou POSTGRES_DB est vide, ou si
        POSTGRES_PORT n'est pas un entier.
    """
    environ = os.environ if environ is None else environ
    missing = [name for name in REQUIRED_VARIABLES if not environ.get(name)]
    if missing:
        raise DatabaseNotConfigured(f"Variables PostgreSQL non renseignées : {', '.join(missing)}.")
    try:
        port = int(environ.get("POSTGRES_PORT") or 5432)
    except ValueError:
        raise DatabaseNotConfigured("POSTGRES_PORT doit être un entier.") from None
    return {
        "host": environ.get("POSTGRES_HOST") or "localhost",
        "port": port,
        "dbname": environ["POSTGRES_DB"],
        "user": environ["POSTGRES_USER"],
        "password": environ["POSTGRES_PASSWORD"],
        "connect_timeout": CONNECT_TIMEOUT_SECONDS,
    }


def connect(environ: Mapping[str, str] | None = None) -> psycopg.Connection:
    """Ouvre une connexion. Lève DatabaseNotConfigured ou psycopg.Error."""
    return psycopg.connect(**connection_params(environ))


def database_is_up(environ: Mapping[str, str] | None = None) -> bool:
    """Vrai si PostgreSQL répond à SELECT 1. Ne lève jamais d'exception.

    Seul le type de l'erreur est journalisé (niveau DEBUG) : le message de psycopg peut
    nommer l'hôte ou l'utilisateur, mais le mot de passe n'apparaît jamais ici.
    """
    try:
        with connect(environ) as connection:
            connection.execute("SELECT 1")
        return True
    except DatabaseNotConfigured as error:
        logger.debug("Base non configurée : %s", error)
        return False
    except Exception as error:  # noqa: BLE001 - /health ne doit jamais échouer
        logger.debug("Base injoignable (%s).", type(error).__name__)
        return False

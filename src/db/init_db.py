"""Création de la table inferences (src/db/schema.sql), tâche A2.

create_schema() est appelée au démarrage du backend (lifespan du routeur predict) et par la
commande :

    uv run python -m src.db.init_db

Le script SQL est idempotent : le relancer ne touche pas une table qui existe déjà. Un verrou
consultatif PostgreSQL sérialise les créations simultanées (plusieurs processus qui démarrent
en même temps), car CREATE TABLE IF NOT EXISTS n'est pas sûr en accès concurrent.
"""

import logging
import sys
from pathlib import Path

import psycopg

from src.db.connection import DatabaseNotConfigured, connect

logger = logging.getLogger("neuraxion.db")

SCHEMA_PATH = Path(__file__).resolve().with_name("schema.sql")
# Clé du verrou consultatif propre à Neuraxion (entier 64 bits quelconque, fixé une fois pour toutes).
SCHEMA_LOCK_KEY = 4_815_162_342


def create_schema(connection: psycopg.Connection | None = None) -> None:
    """Exécute src/db/schema.sql dans une transaction.

    Paramètres
        connection : connexion ouverte à réutiliser ; sans elle, une connexion est ouverte à
            partir des variables POSTGRES_* puis refermée.

    Lève
        DatabaseNotConfigured si les variables de connexion manquent, psycopg.Error si
        PostgreSQL ne répond pas ou refuse le script.
    """
    script = SCHEMA_PATH.read_text(encoding="utf-8")
    if connection is None:
        with connect() as own_connection:
            _run(own_connection, script)
        return
    _run(connection, script)


def _run(connection: psycopg.Connection, script: str) -> None:
    with connection.transaction():
        # Verrou relâché à la fin de la transaction.
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (SCHEMA_LOCK_KEY,))
        # Plusieurs instructions sans paramètre : psycopg les envoie en une seule requête simple.
        connection.execute(script)


def main() -> int:
    """Point d'entrée de python -m src.db.init_db. Renvoie le code de sortie."""
    try:
        create_schema()
    except DatabaseNotConfigured as error:
        print(f"Schéma non créé : {error}", file=sys.stderr)
        return 1
    except psycopg.Error as error:
        # Le message de psycopg peut nommer l'hôte ou l'utilisateur : seul son type est affiché.
        print(
            f"Schéma non créé : PostgreSQL injoignable ou script refusé ({type(error).__name__}). "
            "Vérifiez POSTGRES_HOST et POSTGRES_PORT, et que la base est démarrée.",
            file=sys.stderr,
        )
        return 1
    print("Table inferences prête (src/db/schema.sql).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

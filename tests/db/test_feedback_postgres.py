"""Tests de POST /feedback (tâche F1) contre un vrai PostgreSQL.

Configuration et isolation : tests/db/conftest.py (variables POSTGRES_*, schéma temporaire
neuraxion_test_... imposé par PGOPTIONS, test sauté sans base, en échec si la base est
injoignable). Les tests sans base sont dans tests/api/test_feedback.py.
"""

import re
import secrets
import threading
import time
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql

from src.api import main
from src.db import init_db

DOC_PATH = main.ROOT / "docs" / "api.md"
FREE_TEXT = "le client a payé, secret-7f3a"

# Table telle que la créait src/db/schema.sql avant F1 (sans la colonne feedback).
TABLE_BEFORE_F1 = init_db.SCHEMA_PATH.read_text(encoding="utf-8").split("-- Tâche F1")[0]


def documented_grant(role: str) -> str:
    """GRANT donné par docs/api.md (section POST /feedback), appliqué au rôle demandé."""
    section = DOC_PATH.read_text(encoding="utf-8").split("## POST /feedback")[1].split("\n## ")[0]
    (grant,) = re.findall(r"`(GRANT [^`]+ TO <rôle>)`", section)
    return grant.replace("<rôle>", role)


def feedback_of(database, inference_id) -> str | None:
    (row,) = database.rows("id = %s", (inference_id,))
    return row["feedback"]


def column_names(database) -> list[str]:
    # Colonnes de la table du schéma de test seulement : la base de Compose a aussi la sienne.
    with database.connect() as conn:
        found = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = 'inferences' ORDER BY ordinal_position"
        ).fetchall()
    return [name for (name,) in found]


def predict(client, valid_request) -> str:
    response = client.post("/predict", json=valid_request)
    assert response.status_code == 200, response.text
    return response.json()["inference_id"]


@pytest.fixture(autouse=True)
def private_schema_lock_key(monkeypatch):
    """Clé du verrou consultatif propre à chaque test.

    La clé de init_db vaut pour toute la base : un test qui la garde 5 s gênerait un autre
    lancement de tests/db (ou une API) sur la même base.
    """
    monkeypatch.setattr(init_db, "SCHEMA_LOCK_KEY", secrets.randbelow(2**62) + 1)


@pytest.fixture
def served(registry):
    """Un @champion servi : POST /predict répond 200."""
    return registry.serve()


class LockHolder:
    """Autre session qui garde un verrou tant que la transaction reste ouverte.

    Un minuteur ferme la connexion au bout de 15 s si lock_timeout ne joue pas : le test
    échoue alors sur la durée mesurée au lieu de rester bloqué.
    """

    def __init__(self, database, *queries):
        self.conn = database.connect()
        for query, *parameters in queries:
            self.conn.execute(query, *parameters)
        self.timer = threading.Timer(15, self.conn.close)
        self.timer.start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.timer.cancel()
        self.conn.close()
        return False


# --- schéma ----------------------------------------------------------------------------------


def test_create_schema_adds_the_column_and_its_check_idempotently(database):
    for _ in range(3):
        init_db.create_schema()
    names = column_names(database)
    assert names[-1] == "feedback"
    assert names.count("feedback") == 1
    with database.connect() as conn:
        checks = conn.execute(
            "SELECT conname FROM pg_constraint "
            "WHERE conrelid = 'inferences'::regclass AND conname = 'inferences_feedback_values'"
        ).fetchall()
    assert len(checks) == 1


def test_table_created_before_f1_is_completed_and_keeps_its_rows(database):
    with database.connect() as conn:
        conn.execute(TABLE_BEFORE_F1)
        conn.execute(
            "INSERT INTO inferences (input_data, status, error_message, latency_ms) "
            "VALUES ('{}'::jsonb, 'error', 'ancienne ligne', 1)"
        )
    assert "feedback" not in column_names(database)
    init_db.create_schema()
    assert "feedback" in column_names(database)
    (row,) = database.rows()
    assert (row["error_message"], row["feedback"]) == ("ancienne ligne", None)


# --- route -----------------------------------------------------------------------------------


def test_feedback_after_a_prediction_is_stored_then_replaced(make_app, served, database, valid_request):
    with TestClient(make_app()) as client:
        inference_id = predict(client, valid_request)
        assert feedback_of(database, inference_id) is None
        first = client.post("/feedback", json={"inference_id": inference_id, "feedback": "agree"})
        assert first.status_code == 200, first.text
        assert first.json() == {"inference_id": inference_id, "feedback": "agree"}
        assert feedback_of(database, inference_id) == "agree"
        second = client.post("/feedback", json={"inference_id": inference_id, "feedback": "disagree"})
        assert second.status_code == 200
    assert feedback_of(database, inference_id) == "disagree"


def test_unknown_inference_answers_404_and_changes_nothing(make_app, served, database, valid_request):
    with TestClient(make_app()) as client:
        inference_id = predict(client, valid_request)
        response = client.post("/feedback", json={"inference_id": str(uuid4()), "feedback": "agree"})
    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"
    assert feedback_of(database, inference_id) is None


def test_inference_in_error_has_no_decision_and_answers_404(make_app, database, valid_request):
    # Pas de champion servi : POST /predict répond 503 et enregistre une ligne error.
    with TestClient(make_app()) as client:
        assert client.post("/predict", json=valid_request).status_code == 503
        (row,) = database.rows("status = 'error'")
        response = client.post("/feedback", json={"inference_id": str(row["id"]), "feedback": "disagree"})
    assert response.status_code == 404
    assert feedback_of(database, row["id"]) is None


def test_free_text_is_refused_by_the_api_and_by_the_database(make_app, served, database, valid_request):
    with TestClient(make_app()) as client:
        inference_id = predict(client, valid_request)
        response = client.post("/feedback", json={"inference_id": inference_id, "feedback": FREE_TEXT})
    assert response.status_code == 422
    assert feedback_of(database, inference_id) is None
    with database.connect() as conn, pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("UPDATE inferences SET feedback = %s WHERE id = %s", (FREE_TEXT, inference_id))


def test_column_dropped_after_startup_is_added_back_by_the_route(make_app, served, database, valid_request):
    with TestClient(make_app()) as client:
        inference_id = predict(client, valid_request)
        with database.connect() as conn:
            conn.execute("ALTER TABLE inferences DROP COLUMN feedback")
        response = client.post("/feedback", json={"inference_id": inference_id, "feedback": "agree"})
    assert response.status_code == 200, response.text
    assert feedback_of(database, inference_id) == "agree"


def test_database_absent_at_startup_then_back(make_app, database, monkeypatch):
    monkeypatch.setenv("POSTGRES_PORT", "1")  # port fermé au démarrage : ni table ni colonne
    with TestClient(make_app()) as client:
        down = client.post("/feedback", json={"inference_id": str(uuid4()), "feedback": "agree"})
        monkeypatch.setenv("POSTGRES_PORT", database.environ.get("POSTGRES_PORT", "5432"))
        # La table n'existe pas encore : la route la crée, puis répond 404 pour un id inconnu.
        missing = client.post("/feedback", json={"inference_id": str(uuid4()), "feedback": "agree"})
    assert down.status_code == 503
    assert down.json()["code"] == "DATABASE_UNAVAILABLE"
    assert missing.status_code == 404
    assert "feedback" in column_names(database)


# --- droits ----------------------------------------------------------------------------------


@pytest.fixture
def limited_role(database):
    """Rôle temporaire neuraxion_feedback_limitee_... sans aucun droit, supprimé à la fin.

    Il reçoit seulement USAGE sur le schéma de test : le schéma public n'est pas touché.
    """
    role, password = f"neuraxion_feedback_limitee_{secrets.token_hex(4)}", secrets.token_urlsafe(16)
    with database.connect() as conn:
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(sql.Identifier(role), sql.Literal(password)))
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(database.schema), sql.Identifier(role))
        )
    try:
        yield role, password
    finally:
        with database.connect() as conn:
            conn.execute(sql.SQL("DROP OWNED BY {}").format(sql.Identifier(role)))
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(role)))


def test_role_with_only_the_documented_rights_can_record_feedback(
    make_app, served, database, valid_request, limited_role, monkeypatch
):
    with TestClient(make_app()) as client:
        inference_id = predict(client, valid_request)
    role, password = limited_role
    with database.connect() as conn:
        conn.execute(documented_grant(role))  # le GRANT de docs/api.md, tel quel
        rights = conn.execute(
            "SELECT has_table_privilege(%s, 'inferences', 'UPDATE'), has_table_privilege(%s, 'inferences', 'INSERT')",
            (role, role),
        ).fetchone()
        columns = conn.execute(
            "SELECT column_name, privilege_type FROM information_schema.column_privileges "
            "WHERE grantee = %s AND table_schema = current_schema() AND table_name = 'inferences'",
            (role,),
        ).fetchall()
    assert rights == (False, False)  # droits sur des colonnes seulement, aucun droit sur la table entière
    assert set(columns) == {("id", "SELECT"), ("status", "SELECT"), ("feedback", "UPDATE")}
    monkeypatch.setenv("POSTGRES_USER", role)
    monkeypatch.setenv("POSTGRES_PASSWORD", password)
    with TestClient(make_app()) as client:
        response = client.post("/feedback", json={"inference_id": inference_id, "feedback": "disagree"})
        unknown = client.post("/feedback", json={"inference_id": str(uuid4()), "feedback": "agree"})
    assert response.status_code == 200, response.text
    assert unknown.status_code == 404
    assert feedback_of(database, inference_id) == "disagree"


def test_role_with_the_documented_rights_cannot_add_a_missing_column(
    make_app, served, database, valid_request, limited_role, monkeypatch
):
    with TestClient(make_app()) as client:
        inference_id = predict(client, valid_request)
    role, password = limited_role
    with database.connect() as conn:
        conn.execute(documented_grant(role))
        conn.execute("ALTER TABLE inferences DROP COLUMN feedback")
    monkeypatch.setenv("POSTGRES_USER", role)
    monkeypatch.setenv("POSTGRES_PASSWORD", password)
    with TestClient(make_app(), raise_server_exceptions=False) as client:
        response = client.post("/feedback", json={"inference_id": inference_id, "feedback": "agree"})
    # Comportement documenté : compléter le schéma demande d'être propriétaire de la table.
    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert "feedback" not in column_names(database)


# --- attentes de verrou ----------------------------------------------------------------------


def test_create_schema_does_not_wait_more_than_its_lock_timeout(database):
    init_db.create_schema()
    # Lecture restée ouverte : ACCESS SHARE, incompatible avec le verrou exclusif de l'ALTER TABLE.
    with LockHolder(database, ("SELECT 1 FROM inferences LIMIT 1",)):
        # Connexion ouverte avant la mesure : seule l'attente du verrou est chronométrée (avec
        # POSTGRES_HOST=localhost, Windows perd environ 2 s par connexion à essayer ::1).
        with database.connect() as conn:
            started = time.monotonic()
            with pytest.raises(psycopg.errors.LockNotAvailable):
                init_db.create_schema(conn)
            elapsed = time.monotonic() - started
    assert 4.5 <= elapsed < 6.5, elapsed  # lock_timeout de 5 s


def test_advisory_lock_held_elsewhere_does_not_block_create_schema(database):
    # Une autre création bloquée garde le verrou consultatif : l'attente est bornée elle aussi.
    with LockHolder(database, ("SELECT pg_advisory_lock(%s)", (init_db.SCHEMA_LOCK_KEY,))):
        # Connexion ouverte avant la mesure : seule l'attente du verrou est chronométrée (avec
        # POSTGRES_HOST=localhost, Windows perd environ 2 s par connexion à essayer ::1).
        with database.connect() as conn:
            started = time.monotonic()
            with pytest.raises(psycopg.errors.LockNotAvailable):
                init_db.create_schema(conn)
            elapsed = time.monotonic() - started
    assert 4.5 <= elapsed < 6.5, elapsed  # lock_timeout de 5 s


def test_locked_row_answers_503_within_the_route_timeout_then_200(make_app, served, database, valid_request):
    with TestClient(make_app()) as client:
        inference_id = predict(client, valid_request)
        update = ("UPDATE inferences SET feedback = 'agree' WHERE id = %s", (inference_id,))
        with LockHolder(database, update):  # écriture non validée : la ligne reste verrouillée
            started = time.monotonic()
            locked = client.post("/feedback", json={"inference_id": inference_id, "feedback": "disagree"})
            elapsed = time.monotonic() - started
        assert locked.status_code == 503, locked.text
        assert locked.json()["code"] == "DATABASE_UNAVAILABLE"
        # lock_timeout de 2 s de la route, plus l'ouverture de sa connexion (jusqu'à 2 s de plus
        # avec POSTGRES_HOST=localhost sous Windows) ; le délai de 5 s du schéma en est exclu.
        assert 1.8 <= elapsed < 4.8, elapsed
        assert feedback_of(database, inference_id) is None  # l'écriture de l'autre session a été annulée
        released = client.post("/feedback", json={"inference_id": inference_id, "feedback": "disagree"})
    assert released.status_code == 200
    assert feedback_of(database, inference_id) == "disagree"

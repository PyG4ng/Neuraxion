"""Enregistrement des inférences dans la table inferences (tâche A2) et de leur feedback (F1).

Chaque inférence validée est enregistrée, réussie (status = 'success') ou en erreur
(status = 'error'), avec sa durée. Une requête refusée par la validation (422) ne l'est pas.
Les règles de cohérence (décision qui suit le seuil, ligne success complète, message d'erreur
seulement en erreur) sont garanties par les contraintes CHECK de src/db/schema.sql.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

from src.db.connection import connect

Status = Literal["success", "error"]
FEEDBACK_VALUES = ("agree", "disagree")

INSERT_INFERENCE = """
INSERT INTO inferences (
    id, created_at, input_data, prediction, decision, decision_threshold,
    model_name, model_version, status, error_message, latency_ms
) VALUES (
    %(id)s, %(created_at)s, %(input_data)s, %(prediction)s, %(decision)s, %(decision_threshold)s,
    %(model_name)s, %(model_version)s, %(status)s, %(error_message)s, %(latency_ms)s
)
"""

# Seule une inférence réussie porte une décision à laquelle on peut être d'accord ou non.
# Droits nécessaires : SELECT sur id et status, UPDATE sur feedback (pas de RETURNING).
UPDATE_FEEDBACK = """
UPDATE inferences SET feedback = %(feedback)s
WHERE id = %(id)s AND status = 'success'
"""


@dataclass(frozen=True)
class InferenceRecord:
    """Une ligne de la table inferences.

    input_data ne contient que les variables du contrat de données : c'est à l'appelant de
    le garantir (POST /predict ne garde que les champs de la classe de requête).
    """

    id: UUID
    created_at: datetime
    input_data: Mapping[str, Any]
    status: Status
    latency_ms: int
    prediction: float | None = None
    decision: Literal["approve", "reject"] | None = None
    decision_threshold: float | None = None
    model_name: str | None = None
    model_version: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at doit porter un fuseau horaire.")
        if self.status not in ("success", "error"):
            raise ValueError("status vaut success ou error.")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, int) or self.latency_ms < 0:
            raise ValueError("latency_ms est un entier positif ou nul.")
        for name in ("prediction", "decision_threshold"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and 0 <= value <= 1):
                raise ValueError(f"{name} doit être un nombre fini entre 0 et 1.")

    def as_parameters(self) -> dict[str, Any]:
        """Paramètres nommés de INSERT_INFERENCE."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "input_data": Jsonb(dict(self.input_data)),
            "prediction": self.prediction,
            "decision": self.decision,
            "decision_threshold": self.decision_threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status,
            "error_message": self.error_message,
            "latency_ms": self.latency_ms,
        }


def save_inference(record: InferenceRecord, connection: psycopg.Connection | None = None) -> UUID:
    """Insère une inférence et renvoie son identifiant (record.id).

    Paramètres
        record : la ligne à insérer ; son id est celui renvoyé au client.
        connection : connexion ouverte à réutiliser (la transaction reste à valider par
            l'appelant) ; sans elle, une connexion est ouverte, validée puis refermée.

    Lève
        DatabaseNotConfigured si les variables POSTGRES_* manquent, psycopg.Error si
        PostgreSQL ne répond pas ou refuse la ligne (contrainte CHECK).
    """
    if connection is None:
        with connect() as own_connection:
            own_connection.execute(INSERT_INFERENCE, record.as_parameters())
        return record.id
    connection.execute(INSERT_INFERENCE, record.as_parameters())
    return record.id


def save_feedback(inference_id: UUID, feedback: str, connection: psycopg.Connection | None = None) -> bool:
    """Enregistre agree ou disagree sur une inférence réussie ; remplace un feedback précédent.

    Paramètres
        inference_id : identifiant renvoyé par POST /predict.
        feedback : agree ou disagree.
        connection : connexion ouverte à réutiliser (la transaction reste à valider par
            l'appelant) ; sans elle, une connexion est ouverte, validée puis refermée.

    Renvoie
        True si la ligne a été mise à jour, False si aucune inférence réussie ne porte cet id
        (inférence inconnue, ou en erreur et donc sans décision).

    Lève
        ValueError pour une autre valeur que agree ou disagree, DatabaseNotConfigured si les
        variables POSTGRES_* manquent, psycopg.Error si PostgreSQL ne répond pas, si la colonne
        feedback manque (UndefinedColumn) ou si le rôle n'a pas les droits.
    """
    if feedback not in FEEDBACK_VALUES:
        raise ValueError("feedback vaut agree ou disagree.")
    parameters = {"id": inference_id, "feedback": feedback}
    if connection is None:
        with connect() as own_connection:
            return own_connection.execute(UPDATE_FEEDBACK, parameters).rowcount == 1
    return connection.execute(UPDATE_FEEDBACK, parameters).rowcount == 1

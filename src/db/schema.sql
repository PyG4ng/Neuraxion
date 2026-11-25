-- Table des inférences (§9 du cahier des charges, contrat C4).
-- Idempotent : create_schema() l'exécute à chaque démarrage du backend (tâche A2).
-- Une requête refusée par la validation (422) n'est pas enregistrée ; une inférence validée
-- qui échoue ensuite l'est, avec status = 'error'.
-- La colonne feedback (tâche F1) est ajoutée à la fin, par ALTER TABLE, pour compléter aussi une
-- table créée avant F1.

-- ALTER TABLE prend un verrou exclusif à chaque passage, même si la colonne existe : une
-- transaction restée ouverte sur la table ne bloque pas le démarrage plus de 5 s (échec
-- journalisé, schéma retenté plus tard). SET LOCAL ne vaut que pour la transaction de create_schema().
SET LOCAL lock_timeout = '5s';

CREATE TABLE IF NOT EXISTS inferences (
    id                 uuid             PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at         timestamptz      NOT NULL DEFAULT now(),
    input_data         jsonb            NOT NULL,
    prediction         double precision,
    decision           text,
    decision_threshold double precision,
    model_name         text,
    model_version      text,
    status             text             NOT NULL,
    error_message      text,
    latency_ms         integer,

    CONSTRAINT inferences_input_is_object CHECK (jsonb_typeof(input_data) = 'object'),
    CONSTRAINT inferences_prediction_range CHECK (prediction BETWEEN 0 AND 1),
    CONSTRAINT inferences_decision_values CHECK (decision IN ('approve', 'reject')),
    CONSTRAINT inferences_threshold_range CHECK (decision_threshold BETWEEN 0 AND 1),
    CONSTRAINT inferences_status_values CHECK (status IN ('success', 'error')),
    CONSTRAINT inferences_latency_positive CHECK (latency_ms >= 0),
    CONSTRAINT inferences_success_is_complete CHECK (
        status <> 'success'
        OR (prediction IS NOT NULL AND decision IS NOT NULL AND decision_threshold IS NOT NULL
            AND model_name IS NOT NULL AND model_version IS NOT NULL)
    ),
    CONSTRAINT inferences_decision_matches_threshold CHECK (
        decision IS NULL OR prediction IS NULL OR decision_threshold IS NULL
        OR decision = CASE WHEN prediction >= decision_threshold THEN 'reject' ELSE 'approve' END
    ),
    CONSTRAINT inferences_error_message_only_on_error CHECK (
        (status = 'error' AND error_message IS NOT NULL AND length(btrim(error_message)) > 0)
        OR (status = 'success' AND error_message IS NULL)
    )
);

CREATE INDEX IF NOT EXISTS inferences_created_at_idx ON inferences (created_at);
CREATE INDEX IF NOT EXISTS inferences_model_version_idx ON inferences (model_name, model_version);

-- Tâche F1 : accord (agree) ou désaccord (disagree) avec la décision, NULL tant que rien n'est dit.
-- IF NOT EXISTS rend l'instruction idempotente : si la colonne existe, la contrainte n'est pas
-- recréée non plus. Le feedback n'est jamais utilisé pour l'entraînement (décision H14).
ALTER TABLE inferences ADD COLUMN IF NOT EXISTS feedback text
    CONSTRAINT inferences_feedback_values CHECK (feedback IN ('agree', 'disagree'));

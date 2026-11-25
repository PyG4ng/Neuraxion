# Contrat de l'API

Ce document décrit ce que l'API reçoit et renvoie, et comment chaque inférence est conservée dans PostgreSQL. Les modèles Pydantic correspondants sont dans `src/api/schemas.py` et la table dans `src/db/schema.sql`. Les noms des champs sont en anglais, les messages en français.

Chaque exemple JSON précédé d'un commentaire `exemple` est vérifié par `tests/api/test_contrat_api.py` : si le contrat change, le test échoue tant que ce document n'est pas mis à jour.

## État des routes

| Route | Rôle | Tâche qui l'implémente | Réponse actuelle |
|---|---|---|---|
| `POST /predict` | prédire le risque de défaut d'une demande | A2 | 501 |
| `POST /train` | entraîner, puis promouvoir si le seuil est atteint | A3 | 501 |
| `GET /health` | état de l'API, de la base et du modèle | A1 | route absente |
| `GET /schema` | contrat de données pour le formulaire | A1 | route absente |
| `POST /feedback` | accord ou désaccord avec une décision (facultatif) | F1 | 501 |

Les routes encore vides répondent 501 au format d'erreur commun, en nommant la tâche qui les remplira. Les erreurs de validation (422) prendront le format commun avec le gestionnaire d'erreurs de A1 ; d'ici là, FastAPI renvoie son format par défaut.

## Format d'erreur commun

Toutes les erreurs ont la même forme. Le client teste `code`, qui est stable, plutôt que `detail`, qui peut être reformulé. `errors` n'est rempli que pour une erreur de validation.

<!-- exemple: ErrorResponse -->
```json
{
  "detail": "La demande contient des valeurs non permises.",
  "code": "VALIDATION_ERROR",
  "errors": [
    {"field": "annual_inc", "message": "doit être supérieur ou égal à 0"},
    {"field": "nickname", "message": "champ inconnu"}
  ]
}
```

| Code | Statut HTTP | Cas |
|---|---|---|
| `VALIDATION_ERROR` | 422 | corps invalide : type, borne, valeur non permise, champ manquant ou inconnu |
| `UNAUTHORIZED` | 401 | `POST /train` sans en-tête `X-API-Key` |
| `FORBIDDEN` | 403 | clé fausse, ou `TRAIN_API_KEY` non configurée sur le serveur (la route reste fermée) |
| `NOT_FOUND` | 404 | `POST /feedback` sur une inférence inconnue |
| `TRAINING_IN_PROGRESS` | 409 | un entraînement est déjà en cours |
| `INTERNAL_ERROR` | 500 | erreur inattendue, détaillée dans les logs du backend |
| `NOT_IMPLEMENTED` | 501 | route posée par le contrat, pas encore écrite |
| `MODEL_UNAVAILABLE` | 503 | aucun modèle validé, ou colonnes du modèle différentes du contrat de données |
| `DATABASE_UNAVAILABLE` | 503 | PostgreSQL ne répond pas |

## POST /predict

Ouverte, sans clé. Le corps contient les variables du contrat de données (`contracts/features.yaml`, lot P1), et rien d'autre : un champ inconnu est refusé (422). Le modèle de requête n'est pas écrit à la main :

1. au démarrage, A1 lit le contrat et appelle `build_request_model(contract)`, qui produit une classe Pydantic avec les types, bornes, valeurs permises et champs obligatoires ;
2. A1 crée le routeur avec `predict.build_router(classe)` ;
3. la même classe sert à valider la requête, à décrire la route dans OpenAPI (`/docs`) et à répondre à `GET /schema`.

Changer la liste des variables ne demande donc aucun changement de code dans l'API.

Les exemples de ce document utilisent un contrat d'illustration, en attendant le contrat v0 de P1 :

<!-- contrat-illustration -->
```yaml
version: illustration
features:
  - {name: loan_amnt, type: numeric, min: 1000, max: 40000, unit: dollars, label: Montant demandé, example: 12000}
  - {name: annual_inc, type: numeric, min: 0, unit: dollars par an, label: Revenu annuel, example: 55000}
  - {name: dti, type: numeric, min: 0, max: 100, unit: pourcentage, label: Taux d'endettement, required: false, example: 18.4}
  - {name: home_ownership, type: categorical, allowed: [RENT, OWN, MORTGAGE], label: Logement, example: RENT}
```

Requête :

<!-- exemple-requete: predict -->
```json
{
  "loan_amnt": 12000,
  "annual_inc": 55000,
  "dti": 18.4,
  "home_ownership": "RENT"
}
```

Réponse 200 :

<!-- exemple: PredictResponse -->
```json
{
  "inference_id": "6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d",
  "created_at": "2026-11-27T10:15:42.120000+01:00",
  "default_probability": 0.23,
  "decision": "approve",
  "decision_threshold": 0.31,
  "model_name": "credit-default",
  "model_version": "3"
}
```

- `default_probability` est la probabilité de défaut, entre 0 et 1.
- `decision` vaut `reject` si `default_probability >= decision_threshold`, sinon `approve`. Le modèle de réponse refuse une décision qui ne suit pas cette règle.
- `decision_threshold` est le seuil de décision du modèle servi, choisi à l'entraînement en maximisant le profit réalisé sur 2014. Il ne faut pas le confondre avec `promotion_threshold` (voir `POST /train`).
- `model_version` est la version résolue depuis l'alias `@champion` au moment de la prédiction : un entier écrit en texte (`"3"`), jamais l'alias lui-même.

Sans modèle promu, la réponse est 503 :

<!-- exemple: ErrorResponse -->
```json
{
  "detail": "Aucun modèle validé n'est disponible. Lancez POST /train ou consultez les logs du backend.",
  "code": "MODEL_UNAVAILABLE",
  "errors": null
}
```

## POST /train

Protégée par l'en-tête `X-API-Key`, comparé à `TRAIN_API_KEY`. La clé n'apparaît jamais dans les logs. Le corps choisit un préréglage de `config/training.yaml` ; sans corps, ou sans `preset`, c'est `default`.

Requête complète, avec une clé fictive :

```bash
curl -X POST http://localhost:8000/train   -H "X-API-Key: <valeur de TRAIN_API_KEY>"   -H "Content-Type: application/json"   -d '{"preset": "rapide"}'
```

Corps seul :

<!-- exemple: TrainRequest -->
```json
{"preset": "rapide"}
```

| Préréglage | Usage |
|---|---|
| `default` | toute la période d'entraînement |
| `rapide` | 50 000 prêts tirés avec la graine, pour la démonstration |
| `faible` | modèle volontairement faible, pour montrer un refus |

Réponse 200, modèle promu :

<!-- exemple: TrainResponse -->
```json
{
  "status": "promoted",
  "run_id": "a1b2c3d4e5f60718293a4b5c6d7e8f90",
  "metric_name": "roc_auc",
  "metric_value": 0.71,
  "promotion_threshold": 0.65,
  "promoted": true,
  "model_version": "4",
  "decision_threshold": 0.29
}
```

Réponse 200, modèle refusé : l'alias `@champion` ne bouge pas, le refus est tracé dans MLflow et dans les logs.

<!-- exemple: TrainResponse -->
```json
{
  "status": "rejected",
  "run_id": "0f9e8d7c6b5a49382716f5e4d3c2b1a0",
  "metric_name": "roc_auc",
  "metric_value": 0.58,
  "promotion_threshold": 0.65,
  "promoted": false,
  "model_version": null,
  "decision_threshold": 0.5
}
```

La décision de promotion est prise par `should_promote()` (lot P3) : pour la ROC-AUC, le modèle est promu si `metric_value >= promotion_threshold`. La réponse ne refait pas ce calcul, mais son modèle Pydantic vérifie sa cohérence interne :

- `status` vaut `promoted` si et seulement si `promoted` est vrai ;
- `model_version` est renseigné si et seulement si le modèle est promu, et c'est alors un entier écrit en texte.

Erreurs propres à la route : 401 sans clé, 403 avec une mauvaise clé ou une clé non configurée, 409 si un entraînement est en cours, 422 pour un préréglage inconnu.

## GET /health

Requête : `GET /health`, sans corps ni clé. Elle répond toujours 200 tant que l'API tourne, pour que l'interface puisse afficher l'état. `status` vaut `ok` seulement si la base répond et qu'un modèle est chargé.

<!-- exemple: HealthResponse -->
```json
{
  "status": "ok",
  "database": "ok",
  "model": {"name": "credit-default", "version": "3", "decision_threshold": 0.31}
}
```

Avant la première promotion :

<!-- exemple: HealthResponse -->
```json
{"status": "degraded", "database": "ok", "model": null}
```

## GET /schema

Requête : `GET /schema`, sans corps ni clé. La réponse donne le JSON Schema de la requête de `POST /predict`, produit par la même classe que la validation, et des demandes complètes tirées de `contracts/examples.json`. L'interface génère le formulaire à partir de `request_schema` et propose un bouton par exemple. Chaque exemple est une demande valide pour ce schéma.

Réponse 200 avec le contrat d'illustration :

<!-- exemple: SchemaResponse -->
```json
{
  "contract_version": "illustration",
  "request_schema": {
    "additionalProperties": false,
    "properties": {
      "loan_amnt": {
        "description": "Unité : dollars",
        "examples": [
          12000
        ],
        "maximum": 40000,
        "minimum": 1000,
        "title": "Montant demandé",
        "type": "number"
      },
      "annual_inc": {
        "description": "Unité : dollars par an",
        "examples": [
          55000
        ],
        "minimum": 0,
        "title": "Revenu annuel",
        "type": "number"
      },
      "dti": {
        "anyOf": [
          {
            "maximum": 100,
            "minimum": 0,
            "type": "number"
          },
          {
            "type": "null"
          }
        ],
        "default": null,
        "description": "Unité : pourcentage",
        "examples": [
          18.4
        ],
        "title": "Taux d'endettement"
      },
      "home_ownership": {
        "enum": [
          "RENT",
          "OWN",
          "MORTGAGE"
        ],
        "examples": [
          "RENT"
        ],
        "title": "Logement",
        "type": "string"
      }
    },
    "required": [
      "loan_amnt",
      "annual_inc",
      "home_ownership"
    ],
    "title": "PredictRequest",
    "type": "object"
  },
  "examples": {
    "prudent": {
      "loan_amnt": 8000,
      "annual_inc": 90000,
      "dti": 9.5,
      "home_ownership": "MORTGAGE"
    },
    "moyen": {
      "loan_amnt": 12000,
      "annual_inc": 55000,
      "dti": 18.4,
      "home_ownership": "RENT"
    },
    "risque": {
      "loan_amnt": 30000,
      "annual_inc": 32000,
      "home_ownership": "RENT"
    }
  }
}
```

## POST /feedback (facultatif)

Un choix fermé, sans texte libre, rattaché à une inférence existante. Le feedback ne sert pas à l'entraînement : l'issue réelle d'un prêt n'est connue qu'après 36 mois.

<!-- exemple: FeedbackRequest -->
```json
{"inference_id": "6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d", "feedback": "disagree"}
```

<!-- exemple: FeedbackResponse -->
```json
{"inference_id": "6f1c2a7e-3b4d-4c8a-9e21-5d7f0a1b2c3d", "feedback": "disagree"}
```

Erreurs : 404 si l'inférence n'existe pas, 422 pour une autre valeur que `agree` ou `disagree`.

## Table inferences

Créée par `src/db/schema.sql`, que le backend exécute à chaque démarrage. Le script est idempotent : il ne touche pas une table qui existe déjà.

| Colonne | Type | Rôle |
|---|---|---|
| `id` | uuid, clé | identifiant de l'inférence, renvoyé dans `inference_id` |
| `created_at` | timestamptz | date et heure de réception de la requête, transmises par A2 (`now()` sert seulement de valeur par défaut) |
| `input_data` | jsonb | variables validées, sans aucun autre champ |
| `prediction` | double precision | probabilité de défaut, entre 0 et 1 |
| `decision` | text | `approve` ou `reject` |
| `decision_threshold` | double precision | seuil de décision utilisé |
| `model_name` | text | modèle enregistré, `credit-default` |
| `model_version` | text | version résolue depuis `@champion` |
| `status` | text | `success` ou `error` |
| `error_message` | text | cause maîtrisée de l'échec, non vide si `status = 'error'`, vide sinon ; jamais un message d'exception brut qui recopierait la demande |
| `latency_ms` | integer | durée du traitement en millisecondes |

Règles :

- une requête refusée par la validation (422) n'est pas enregistrée ;
- une inférence validée qui échoue ensuite (modèle indisponible, erreur du modèle) est enregistrée avec `status = 'error'` ;
- si PostgreSQL ne répond pas, l'inférence ne peut pas être enregistrée : l'API répond 503 (`DATABASE_UNAVAILABLE`) et l'écrit dans les logs, sans les données de la demande. Il n'y a pas de reprise différée ;
- une ligne `success` a forcément une prédiction, une décision, un seuil, un nom et une version de modèle (contrainte `CHECK`) ;
- la décision enregistrée suit le seuil : `reject` si `prediction >= decision_threshold`, sinon `approve` (contrainte `CHECK`) ;
- une ligne `error` garde les valeurs déjà connues (version du modèle, par exemple) et laisse vides les autres ;
- ni adresse IP ni navigateur ne sont enregistrés, et les logs ne contiennent pas les données de la demande ;
- la colonne `feedback` sera ajoutée par la tâche F1, avec `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`.

Pour retrouver une inférence et la version du modèle :

```sql
SELECT id, created_at, prediction, decision, model_version
FROM inferences
ORDER BY created_at DESC
LIMIT 5;
```

## Points à confirmer à la revue des contrats (C6)

- Le format de `contracts/features.yaml` (lot P1). `build_request_model()` suppose pour chaque variable : `name` et `type` (`numeric`, `integer` ou `categorical`) obligatoires ; `min`, `max`, `allowed`, `required` (vrai par défaut), `label`, `unit` et `example` facultatifs.
- Les valeurs `agree` et `disagree` du feedback.
- Le raccordement A1 → A2 : A1 construit la classe de requête et appelle `predict.build_router()` ; A2 remplit le corps de la route sans changer sa signature.
- Les choix propres à C4 : codes d'erreur supplémentaires, `GET /health` toujours en 200 avec `ok` ou `degraded`.

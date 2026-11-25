# Contrat de l'API

Ce document décrit ce que l'API reçoit et renvoie, et comment chaque inférence est conservée dans PostgreSQL. Les modèles Pydantic correspondants sont dans `src/api/schemas.py` et la table dans `src/db/schema.sql`. Les noms des champs sont en anglais, les messages en français.

Chaque exemple JSON précédé d'un commentaire `exemple` est vérifié par `tests/api/test_contrat_api.py` : si le contrat change, le test échoue tant que ce document n'est pas mis à jour.

## État des routes

| Route | Rôle | Tâche qui l'implémente | Réponse actuelle |
|---|---|---|---|
| `POST /predict` | prédire le risque de défaut d'une demande | A2 | 200 |
| `POST /train` | entraîner, puis promouvoir si le seuil est atteint | A3 | 200 (501 tant que R2 n'est pas livrée) |
| `GET /health` | état de l'API, de la base et du modèle | A1 | 200 |
| `GET /schema` | contrat de données pour le formulaire | A1 | 200 |
| `POST /feedback` | accord ou désaccord avec une décision (facultatif) | F1 | 200 |

Les routes encore vides répondent 501 au format d'erreur commun, en nommant la tâche qui les remplira. Toutes les erreurs, y compris les erreurs de validation (422), les routes inconnues (404) et les erreurs imprévues (500), ont le format commun ci-dessous (`src/api/errors.py`).

## Format d'erreur commun

Toutes les erreurs ont la même forme. Le client teste `code`, qui est stable, plutôt que `detail`, qui peut être reformulé. `errors` n'est rempli que pour une erreur de validation.

<!-- exemple: ErrorResponse -->
```json
{
  "detail": "La demande a été refusée : 2 champs ne respectent pas le contrat. Les champs attendus sont décrits par GET /schema.",
  "code": "VALIDATION_ERROR",
  "errors": [
    {"field": "annual_inc", "message": "doit être supérieur ou égal à 0"},
    {"field": "nickname", "message": "champ inconnu"}
  ]
}
```

| Code | Statut HTTP | Cas |
|---|---|---|
| `VALIDATION_ERROR` | 422 | corps invalide : JSON mal formé, type, borne, valeur non permise, champ manquant ou inconnu |
| `BAD_REQUEST` | 400 | requête mal formée refusée avant la validation (UTF-8 invalide, JSON trop imbriqué), et tout autre statut 4xx non listé ici |
| `UNAUTHORIZED` | 401 | `POST /train` sans en-tête `X-API-Key` |
| `FORBIDDEN` | 403 | clé fausse, ou `TRAIN_API_KEY` non configurée sur le serveur (la route reste fermée) |
| `NOT_FOUND` | 404 | route inconnue, ou `POST /feedback` sur une inférence inconnue ou enregistrée en erreur |
| `METHOD_NOT_ALLOWED` | 405 | méthode non prévue pour la route (`GET /predict`), avec l'en-tête `Allow` |
| `TRAINING_IN_PROGRESS` | 409 | un entraînement est déjà en cours (levé explicitement par A3 ; une autre 409 reçoit `BAD_REQUEST`) |
| `PAYLOAD_TOO_LARGE` | 413 | corps de plus de 64 Ko |
| `INTERNAL_ERROR` | 500 | erreur inattendue, détaillée dans les logs du backend, échec du modèle pendant `POST /predict`, et tout autre statut 5xx non listé ici |
| `NOT_IMPLEMENTED` | 501 | route posée par le contrat, pas encore écrite |
| `MODEL_UNAVAILABLE` | 503 | aucun modèle validé, ou colonnes du modèle différentes du contrat de données |
| `DATABASE_UNAVAILABLE` | 503 | PostgreSQL ne répond pas |

## Démarrage, validation et logs (A1)

Lancement : `uv run uvicorn src.api.main:app`. Au démarrage, `src/api/main.py` :

1. lit `contracts/features.yaml` et construit la classe de requête avec `build_request_model()` ;
2. lit `contracts/examples.json`, un objet dont chaque clé est un profil et chaque valeur une demande complète, et valide chaque exemple avec cette classe ;
3. inclut les routeurs `predict`, `train` et `feedback`, puis `GET /health` et `GET /schema`.

Si l'un des deux fichiers manque, est illisible ou ne respecte pas le contrat, l'API ne démarre pas : uvicorn affiche une seule ligne `Démarrage impossible : ...`, qui nomme le fichier (et le profil fautif pour un exemple). Le contrat est contrôlé variable par variable : nom en lettres, chiffres et `_` (pas de `model_` en tête), `min` et `max` numériques (entiers pour une variable `integer`), `allowed` liste non vide, sans doublon, du bon type et dans les bornes, sans caractère NUL et, pour un nombre, sans valeur au-delà de 2^53 en valeur absolue, `example` conforme à la variable, `required` booléen, `label` et `unit` textes. Un fichier illisible (droits, encodage) est signalé de la même façon. `version` s'écrit entre guillemets (`version: "1.10"`), sinon YAML la lit comme un nombre.

Validation de la requête de `POST /predict` :

- un champ inconnu est refusé (`champ inconnu`) ;
- les types sont stricts : `"12000"` (texte) ou `true` ne sont pas des nombres, et `2.5` comme `36.0` ne sont pas des entiers, y compris pour une variable munie de `allowed` (publiée dans le JSON Schema par `enum`) ;
- `Infinity` et `NaN`, que le décodeur JSON de Python accepte, sont refusés ;
- une variable catégorielle sans `allowed` refuse le caractère NUL (`doit être une chaîne de caractères valide`), que PostgreSQL ne sait pas stocker dans `input_data` ; une variable `integer` refuse une valeur au-delà de 2^53 en valeur absolue, qui pourrait ne pas passer sans perte au modèle ;
- les messages de `errors` sont en français et ne recopient jamais la valeur reçue ; au-delà de 50 erreurs, seules les 50 premières sont listées et `detail` donne le total ;
- un corps de plus de 64 Ko est refusé (413 `PAYLOAD_TOO_LARGE`) : dès l'en-tête s'il annonce sa taille, sinon pendant sa lecture par une route qui lit le corps (une route inconnue répond 404 sans le lire) ;
- une erreur qui porte sur le corps entier (JSON mal formé, corps qui n'est pas un objet) a un `detail` qui le dit et le champ `corps` dans `errors`. Un corps que le décodeur refuse avant la validation (UTF-8 invalide, JSON imbriqué à l'extrême) reçoit 400 `BAD_REQUEST`.

Les routes signalent une erreur maîtrisée en levant `ApiError(status_code, code, detail, headers=None)` (`src/api/errors.py`), par exemple `ApiError(503, ErrorCode.MODEL_UNAVAILABLE, ...)` dans A2 ou `ApiError(401, ErrorCode.UNAUTHORIZED, ...)` dans A3. Une exception imprévue donne une 500 `INTERNAL_ERROR`, sans son message. Une exception levée après le début de la réponse (tâche d'arrière-plan, réponse en flux) est aussi rattrapée et journalisée sans son message.

Logs (`neuraxion.api`, sortie d'erreur) : une ligne par requête avec la méthode (`<méthode inconnue>` si le client en invente une), le modèle de route (`/predict`, `/essai/{code}`, jamais le chemin brut ni les paramètres), le statut et la durée, par exemple `POST /predict -> 422 (3 ms)`. Une demande refusée ajoute une ligne qui nomme les champs de premier niveau déclarés par la route (les variables du contrat pour `/predict`) et compte les autres (champs inconnus, clés de dictionnaire) sans les nommer. Les messages de `errors` ne reprennent que des textes fixes ou les valeurs du contrat : le message d'un validateur personnalisé devient `valeur invalide`. Une erreur interne ajoute le type de l'exception et les cadres de pile situés dans `src/` (fichier, ligne, fonction), sans le message. Le journal d'accès d'uvicorn, qui contient l'adresse IP du client, est coupé à la création de l'application et de nouveau au démarrage ; la commande de démarrage d'I1 peut aussi passer `--no-access-log`.

Raccord avec A2 : le modèle servi est décrit par `app.state.model_info` (`ModelInfo`, un objet qui a les attributs `name`, `version` et `decision_threshold`, ou `None` tant qu'aucun modèle n'est chargé). A2 le renseigne depuis le routeur `predict` (paramètre `lifespan` de l'`APIRouter`, que FastAPI fusionne avec celui de l'application), sans modifier `main.py`. Une valeur mal formée est traitée comme absente par `/health`, avec un avertissement dans les logs. Les tests de A2 doivent ouvrir le client avec `with TestClient(app) as client`, sans quoi le lifespan ne s'exécute pas.

Base de données : `src/db/connection.py` lit `POSTGRES_HOST` (`localhost` par défaut), `POSTGRES_PORT` (5432 par défaut ; 5433 depuis la machine hôte avec Compose), `POSTGRES_USER`, `POSTGRES_PASSWORD` et `POSTGRES_DB`. Sans ces trois dernières variables (ou si l'une est vide), ou si `POSTGRES_PORT` n'est pas un entier, la base est « non configurée » et `/health` répond `database: unavailable` sans tenter de connexion. `/health` répond en 2 s au plus, même si la base ne répond pas, et accepte aussi `HEAD`. Un seul contrôle de la base tourne à la fois, dans un fil démon : les appels qui arrivent pendant ce contrôle l'attendent (au plus 2 s) au lieu d'en lancer d'autres, et un contrôle bloqué n'empêche pas l'arrêt du processus. `database_check` doit être une fonction synchrone qui renvoie `True` si la base répond ; toute autre valeur compte comme `unavailable`. A2 réutilise ce module.

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
- `model_version` est la version résolue depuis l'alias `@champion` au chargement du modèle (démarrage de l'API, ou rechargement après une promotion), puis gardée en mémoire avec le modèle : un entier écrit en texte (`"3"`), jamais l'alias lui-même.

Sans modèle promu, la réponse est 503 :

<!-- exemple: ErrorResponse -->
```json
{
  "detail": "Aucun modèle validé n'est disponible. Lancez POST /train ou consultez les logs du backend.",
  "code": "MODEL_UNAVAILABLE",
  "errors": null
}
```

### Modèle servi et enregistrement (A2)

Le code est dans `src/api/predict.py` (route et démarrage), `src/api/served_model.py` (adaptateur du @champion), `src/db/init_db.py` (`create_schema()`) et `src/db/inferences.py` (`save_inference()`).

Au démarrage, le `lifespan` du routeur `predict` (fusionné par FastAPI avec celui de l'application, sans toucher à `main.py`) :

1. appelle `create_schema()`, qui exécute `src/db/schema.sql` sous un verrou consultatif PostgreSQL (deux processus qui démarrent ensemble ne se gênent pas). Une base absente ou non configurée n'empêche pas le démarrage : un avertissement est journalisé, et la table sera créée par la première inférence qui ne la trouve pas. Chaque inférence tente d'abord l'insertion : un rôle PostgreSQL qui a seulement le droit d'insérer dans une table déjà créée suffit ;
2. charge le @champion une fois, par `load_champion()`, et le garde en mémoire. Les requêtes suivantes ne contactent pas MLflow ;
3. renseigne `app.state.model_info` (lu par `/health`) et pose `app.state.reload_model`.

`app.state.reload_model` (raccord avec A3) : fonction synchrone, sans argument, qui recharge le @champion et met à jour le modèle servi et `app.state.model_info`. Elle renvoie le `ModelInfo` du modèle servi après l'appel, ou `None`. A3 l'appelle après une promotion, depuis une route `def` ou par `run_in_threadpool` depuis une route `async`, car elle bloque le temps du chargement. Deux appels simultanés sont exécutés l'un après l'autre. Comportement :

- nouveau champion valide : il est servi dès le retour de la fonction ;
- plus de champion (`load_champion()` renvoie `None`) : plus aucun modèle n'est servi, `POST /predict` répond 503 ;
- champion qui ne peut pas être servi (variables différentes du contrat, version, seuil ou classes inexploitables) : plus aucun modèle n'est servi, `POST /predict` répond 503 (garde-fou du plan : servir l'ancien modèle donnerait une autre version que `@champion`) ;
- chargement en échec (MLflow injoignable, exception levée par `load_champion()`) : le modèle déjà servi est gardé, l'échec est journalisé, et la valeur renvoyée permet de voir que la version n'a pas changé.

Aucun délai maximal n'est imposé à `load_champion()` : si MLflow ne répond pas, le démarrage (ou `reload_model()`) attend. Le délai de connexion du client MLflow relève de R1.

Interface attendue de `load_champion()` (tâche R1, `src/registry/champion.py`), appelée sans argument : `None` sans champion, sinon un objet (attributs) ou un dictionnaire (clés) qui fournit :

| Attribut | Contenu |
|---|---|
| `model` | modèle chargé, avec `predict_proba` et `classes_` (par exemple `mlflow.sklearn.load_model`) ; `classes_` contient exactement 0 et 1, et la probabilité de défaut est la colonne de la classe 1 |
| `version` (ou `model_version`) | version résolue depuis l'alias : entier positif, y compris un entier numpy, ou texte de chiffres (`"3"`, espaces autour retirés) |
| `decision_threshold` | seuil de décision entre 0 et 1 : nombre, ou texte numérique comme un tag MLflow (`"0.31"`) |
| `features` (ou `feature_names`) | variables attendues par le modèle : liste, tuple, tableau numpy ou `pandas.Index` (l'adaptateur refuse un dictionnaire ou un ensemble : l'ordre des colonnes doit venir d'une séquence). Les noms sont exactement ceux du contrat de données, dans n'importe quel ordre ; c'est l'ordre du modèle qui est utilisé |
| `name` (ou `model_name`), facultatif | nom du modèle ; sinon `MLFLOW_MODEL_NAME`, puis `credit-default` |

Un attribut qui vaut `None` compte comme absent : le second nom est alors lu (`model_version` si `version` vaut `None`). `load_champion()` doit être une fonction synchrone.

Un champion qui ne respecte pas cette interface (variables différentes du contrat, version qui n'est pas un entier positif, seuil hors de [0, 1], pas de `predict_proba`, `classes_` absent ou différent de 0 et 1) n'est pas servi : `POST /predict` répond 503 `MODEL_UNAVAILABLE` et le log nomme la cause. Tant que le module `src.registry.champion` n'existe pas, il n'y a pas de champion : un avertissement le dit au démarrage.

Le modèle reçoit un `DataFrame` d'une ligne, colonnes dans l'ordre de `features`. Les variables catégorielles sont de type `object`, les autres `float64`, et une variable facultative absente vaut `NaN` : le pipeline du lot P2 doit savoir l'imputer.

Réponses de `POST /predict` et lignes enregistrées :

| Cas | Réponse | Ligne dans `inferences` |
|---|---|---|
| prédiction réussie | 200 `PredictResponse` | `success`, avec la probabilité, la décision, le seuil, le nom et la version |
| entrée refusée par le contrat | 422 `VALIDATION_ERROR` | aucune |
| pas de modèle servi | 503 `MODEL_UNAVAILABLE` | `error`, `Aucun modèle validé n'est disponible.` |
| le modèle lève une exception ou renvoie autre chose qu'une probabilité entre 0 et 1 | 500 `INTERNAL_ERROR` | `error`, avec le nom, la version et le seuil du modèle, et une cause fixe (type de l'exception, jamais son message) |
| PostgreSQL injoignable ou non configurée pendant l'enregistrement d'une prédiction | 503 `DATABASE_UNAVAILABLE` | aucune (la base ne répond pas) |

Si la base ne répond pas pendant l'enregistrement d'une inférence en erreur, la réponse garde la cause première (503 `MODEL_UNAVAILABLE` ou 500) et l'échec de la base est journalisé. Une ligne refusée par une contrainte `CHECK` est une erreur de programmation : 500 `INTERNAL_ERROR`. Une table absente (base injoignable au démarrage, table supprimée depuis) est créée, puis l'insertion est retentée une fois.

`created_at` est l'heure à laquelle la route reçoit la demande validée, en UTC (suffixe `Z`, par exemple `2026-11-27T09:15:42.120000Z` ; l'exemple plus haut montre qu'un autre fuseau est aussi valide pour le contrat). `latency_ms` mesure le traitement jusqu'à la décision, ou jusqu'à l'échec pour une ligne `error`, sans l'écriture en base. Pour un échec du modèle, `error_message` vaut `Le modèle a échoué pendant la prédiction (ValueError).` (le type de l'exception, jamais son message ; un nom de type qui n'est pas un identifiant Python d'au plus 64 caractères devient `exception`) ou `Sortie du modèle inexploitable : ...`. Une sortie de `predict_proba` en tableau `object` faite de réels ou de `Decimal` est convertie en `float64` ; un booléen ou un complexe est refusé. `input_data` contient toutes les variables du contrat et elles seules, `null` pour une variable facultative absente. Logs : chaque requête a sa ligne d'accès (méthode, route, statut, durée, voir A1) ; une inférence réussie ajoute une ligne avec son identifiant, le nom et la version du modèle et la durée ; une erreur du modèle ou une base injoignable ajoute une ligne avec l'identifiant et le type de l'erreur ; sans modèle, seule la ligne d'accès est écrite (l'absence de modèle est signalée au chargement). Jamais les données de la demande, la probabilité ni le message d'une exception.

À raccorder quand R1 (#8) sera livré :

- vérifier que `load_champion()` renvoie les attributs ci-dessus, ou adapter seulement `to_served_model()` dans `src/api/served_model.py` ;
- vérifier que le modèle chargé expose `predict_proba` et `classes_` (`mlflow.sklearn.load_model` le fait, un modèle `pyfunc` non) ;
- lancer `POST /predict` sur le modèle factice de `scripts/register_dummy_model.py` et vérifier la ligne en base (requête SQL ci-dessous) ;
- A3 appelle `app.state.reload_model()` après une promotion.

## POST /train

Protégée par l'en-tête `X-API-Key`, comparé à la variable d'environnement `TRAIN_API_KEY` du backend (`.env.example`). La clé n'apparaît jamais dans les logs ni dans les réponses. Le corps choisit un préréglage de `config/training.yaml` ; sans corps, ou sans `preset`, c'est `default`. La réponse arrive à la fin de l'entraînement : avec `curl`, ne pas fixer de délai maximal trop court.

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

### Contrôles et erreurs (A3)

Le code est dans `src/api/train.py` et `src/api/security.py`. Les contrôles se font dans cet ordre, et le premier qui échoue donne la réponse :

| Ordre | Contrôle | Réponse si refus |
|---|---|---|
| 1 | en-tête `X-API-Key` présent et non vide | 401 `UNAUTHORIZED`, avec l'en-tête `WWW-Authenticate: APIKey` |
| 2 | clé égale à `TRAIN_API_KEY` | 403 `FORBIDDEN`, avec le même message si la clé est fausse ou si `TRAIN_API_KEY` est absente ou vide sur le serveur (la route reste alors fermée) |
| 3 | `train_and_promote` disponible (tâche R2) | 501 `NOT_IMPLEMENTED`, qui nomme R2 et `src/registry/promotion.py` |
| 4 | préréglage connu | 422 `VALIDATION_ERROR` sur le champ `preset` |
| 5 | aucun entraînement en cours | 409 `TRAINING_IN_PROGRESS` |

La clé est vérifiée avant le corps : un client sans la bonne clé ne reçoit que 401 ou 403, quel que soit le préréglage. Seuls les corps que FastAPI ou le middleware de A1 refusent avant la lecture des en-têtes de sécurité y échappent : JSON mal formé (422, sur le champ `corps`), UTF-8 invalide ou JSON trop imbriqué (400) et corps de plus de 64 Ko (413). Ces réponses ne disent rien de la clé ni du préréglage. Plusieurs en-têtes `X-API-Key` dans la même requête sont refusés (403), même si le premier est vide, puisque seul le premier serait lu. Les espaces autour de `TRAIN_API_KEY` sont ignorés ; une clé non ASCII doit être envoyée en UTF-8. La comparaison se fait à temps constant (`hmac.compare_digest` sur les empreintes SHA-256 des deux clés) : sa durée ne dépend ni du nombre de caractères justes ni de la longueur de la clé attendue. Les logs disent seulement pourquoi la requête est refusée (en-tête absent, plusieurs en-têtes, clé fausse, `TRAIN_API_KEY` non configurée). La route ne limite pas le nombre de tentatives : la clé doit donc être longue et tirée au hasard, par exemple avec `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

Un seul entraînement à la fois : la route prend un verrou sans attendre, et une requête qui arrive pendant un entraînement reçoit aussitôt :

<!-- exemple: ErrorResponse -->
```json
{
  "detail": "Un entraînement est déjà en cours. Réessayez quand il sera terminé.",
  "code": "TRAINING_IN_PROGRESS",
  "errors": null
}
```

Le verrou est libéré à la fin de l'entraînement, qu'il réussisse ou échoue, même si le client s'est déconnecté entre-temps. Il vaut pour un processus : l'API doit tourner avec un seul processus uvicorn (sans `--workers`), point à reprendre dans la commande de démarrage du backend (tâche I1). La route est synchrone : FastAPI l'exécute dans un fil, si bien que `/health`, `/schema` et `/predict` continuent de répondre pendant l'entraînement.

Une exception levée par `train_and_promote` ou à l'import de son module, `sys.exit()` et toute autre `BaseException` compris, donne une 500 `INTERNAL_ERROR`, journalisée avec son type et ses cadres dans `src/`, sans son message. Un module introuvable à l'intérieur de R2 est nommé dans les logs. Un résultat qui ne respecte pas `TrainResponse`, ou dont la métrique ou un seuil vaut `NaN` ou l'infini, donne aussi une 500, et les logs nomment les champs fautifs, sans leurs valeurs.

### Rechargement du modèle servi

Après une promotion, la route appelle `app.state.reload_model()`, posée par A2 (#15) dans le `lifespan` du routeur `predict` : une fonction synchrone, sans argument, qui recharge le `@champion` et met à jour `app.state.model_info`. `POST /predict` et `GET /health` donnent alors la nouvelle version. L'appel a lieu avant la libération du verrou, et aussi, par prudence, après un entraînement en échec ou un résultat non conforme, puisque l'alias a pu bouger. Si `reload_model` manque, n'est pas une fonction ou est asynchrone, la réponse reste 200 (l'entraînement a réussi) et un avertissement est journalisé ; si elle lève une exception (`SystemExit` et toute autre `BaseException` comprises), la réponse reste aussi 200 et l'erreur est journalisée avec son type et ses cadres, sans son message. Dans tous ces cas, `/predict` garde le modèle précédent jusqu'au prochain rechargement ou redémarrage.

### Raccord avec R2 (lot P3)

`train_and_promote(preset)` n'existe pas encore (#14). La route l'importe seulement à la demande (`get_train_and_promote`, dans `src/api/train.py`) : l'API démarre sans le module, et sans charger MLflow. Tant que `src/registry/promotion.py` manque, ou qu'il ne définit pas `train_and_promote`, une requête authentifiée reçoit 501 `NOT_IMPLEMENTED`. Ce statut est le même que celui des routes encore vides du contrat : la fonctionnalité n'est pas encore dans le code, ce n'est pas une panne passagère (503, qu'un client ou une sonde de santé pourrait réessayer). Une dépendance manquante à l'intérieur du module de R2 (MLflow absent, par exemple) n'est pas masquée : elle donne une 500 et ses cadres dans les logs.

Ce que la route attend de R2 :

- `train_and_promote(preset)` est synchrone et reçoit `default`, `rapide` ou `faible` en argument positionnel ;
- elle renvoie une `TrainResponse`, un dictionnaire ou un objet qui ont au moins les huit champs de la réponse ci-dessus ; un champ en plus est ignoré, et les valeurs sont validées selon les règles souples de Pydantic (un entier pour un nombre décimal, par exemple) ;
- elle lève une exception si l'entraînement échoue (préréglage absent de `config/training.yaml`, données manquantes) ;
- elle déplace elle-même `@champion` et pose le tag `decision_threshold` ; la route ne touche pas au Model Registry.

Quand R2 sera livrée, il restera à :

1. vérifier que la valeur renvoyée par `train_and_promote` passe `TrainResponse` (un test de bout en bout avec le préréglage `rapide`) ;
2. vérifier avec le vrai `@champion` de R1 que `/predict` renvoie la nouvelle version après une promotion (critère de l'issue #21). Avec un faux `@champion` et le vrai `app.state.reload_model` de A2, c'est déjà testé : `tests/api/test_train.py::test_a_promotion_makes_predict_serve_the_new_version`, et sur PostgreSQL `tests/db/test_api_postgres.py::test_the_recorded_version_follows_a_promotion` ;
3. retirer de ce document la mention « 501 tant que R2 n'est pas livrée » du tableau des routes.

Les tests (`tests/api/test_train.py`) remplacent `train_and_promote` par une fausse fonction (`app.dependency_overrides[train.get_train_and_promote]`) ou par un faux module `src.registry.promotion`.

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

Erreurs : 404 si l'inférence n'existe pas ou a été enregistrée en erreur, 422 pour une autre valeur que `agree` ou `disagree`, 503 si la base ne répond pas.

Comportement (F1, `src/api/feedback.py`) :

- `inference_id` est celui d'une réponse 200 de `POST /predict`. Une inférence enregistrée en erreur n'a pas de décision : elle répond 404 comme une inférence inconnue (`NOT_FOUND`) ;
- le choix est écrit dans la colonne `feedback` de la ligne ; un nouvel appel le remplace ;
- `feedback` n'accepte que `agree` ou `disagree`, en minuscules : tout autre texte, une autre casse, une valeur vide ou d'un autre type est refusé (422 `VALIDATION_ERROR`, champ `feedback`), de même qu'un champ en plus (`champ inconnu`). Un `inference_id` qui n'est pas un UUID est refusé (422, champ `inference_id`). Une valeur refusée n'est recopiée ni dans la réponse ni dans les logs ; la valeur acceptée est renvoyée dans la réponse 200, mais pas journalisée ;
- PostgreSQL injoignable ou non configurée : 503 `DATABASE_UNAVAILABLE`, rien n'est enregistré ;
- si la table ou la colonne manque encore (base absente au démarrage), la route exécute `create_schema()` puis réessaie une fois. Compléter le schéma demande d'être propriétaire de la table : avec les seuls droits ci-dessous, la route répond alors 500 `INTERNAL_ERROR`, et il faut lancer `uv run python -m src.db.init_db` avec le rôle propriétaire ;
- le feedback n'est jamais utilisé pour l'entraînement (décision d'équipe H14) : il sert seulement à suivre l'avis des utilisateurs sur les décisions.

Droits PostgreSQL du rôle de l'API pour cette route : `SELECT` sur `id` et `status`, `UPDATE` sur `feedback`, par exemple `GRANT SELECT (id, status), UPDATE (feedback) ON inferences TO <rôle>`. La colonne doit déjà exister (créée par le rôle propriétaire de la table).

Pour suivre les avis :

```sql
SELECT decision, feedback, count(*)
FROM inferences
WHERE status = 'success'
GROUP BY decision, feedback;
```

## Table inferences

Créée par `src/db/schema.sql`, que le backend exécute à chaque démarrage (`create_schema()`). Pour la créer sans lancer l'API :

```bash
uv run python -m src.db.init_db
```

La commande lit les variables `POSTGRES_*`, affiche `Table inferences prête (src/db/schema.sql).` et sort avec le code 0, ou affiche `Schéma non créé : ...` et sort avec le code 1, sans le message de psycopg. Le script est idempotent : une table qui existe déjà garde ses lignes, et seuls les index et la colonne `feedback` manquants sont créés.

| Colonne | Type | Rôle |
|---|---|---|
| `id` | uuid, clé | identifiant de l'inférence, renvoyé dans `inference_id` |
| `created_at` | timestamptz | date et heure de réception de la requête, transmises par A2 (`now()` sert seulement de valeur par défaut) |
| `input_data` | jsonb | variables validées, sans aucun autre champ |
| `prediction` | double precision | probabilité de défaut, entre 0 et 1 |
| `decision` | text | `approve` ou `reject` |
| `decision_threshold` | double precision | seuil de décision utilisé |
| `model_name` | text | modèle enregistré : nom fourni par `load_champion()`, sinon `MLFLOW_MODEL_NAME`, sinon `credit-default` |
| `model_version` | text | version résolue depuis `@champion` |
| `status` | text | `success` ou `error` |
| `error_message` | text | cause maîtrisée de l'échec, non vide si `status = 'error'`, vide sinon ; jamais un message d'exception brut qui recopierait la demande |
| `latency_ms` | integer | durée du traitement en millisecondes |
| `feedback` | text | `agree` ou `disagree`, posé par `POST /feedback` (F1) ; vide tant qu'aucun avis n'est donné |

Règles :

- une requête refusée par la validation (422) n'est pas enregistrée ;
- une inférence validée qui échoue ensuite (modèle indisponible, erreur du modèle) est enregistrée avec `status = 'error'` ;
- si PostgreSQL ne répond pas (ou n'est pas configurée) au moment d'enregistrer une prédiction calculée, l'API répond 503 (`DATABASE_UNAVAILABLE`) sans renvoyer la prédiction, et l'écrit dans les logs, sans les données de la demande. Il n'y a pas de reprise différée ;
- si c'est une ligne `error` qui ne peut pas être enregistrée, la réponse garde la cause première (503 `MODEL_UNAVAILABLE` ou 500) et la ligne manque ;
- une ligne `success` a forcément une prédiction, une décision, un seuil, un nom et une version de modèle (contrainte `CHECK`) ;
- la décision enregistrée suit le seuil : `reject` si `prediction >= decision_threshold`, sinon `approve` (contrainte `CHECK`) ;
- une ligne `error` garde les valeurs déjà connues (version du modèle, par exemple) et laisse vides les autres ;
- ni adresse IP ni navigateur ne sont enregistrés, et les logs ne contiennent pas les données de la demande ;
- la colonne `feedback` (F1) est ajoutée à la fin de `src/db/schema.sql` par `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, avec sa contrainte `CHECK (feedback IN ('agree', 'disagree'))` : une table créée avant F1 est complétée au démarrage suivant, sans perdre ses lignes. Cet `ALTER TABLE` prend un verrou exclusif bref à chaque passage, limité à 5 s par `lock_timeout` : une transaction restée ouverte sur la table fait échouer la création du schéma (journalisée), sans bloquer le démarrage ;
- seule une ligne `success` peut recevoir un feedback.

Pour retrouver une inférence et la version du modèle :

```sql
SELECT id, created_at, prediction, decision, model_version
FROM inferences
ORDER BY created_at DESC
LIMIT 5;
```

## Tests de l'API et de la base (A4)

Deux dossiers :

- `tests/api` : l'API sans base ni MLflow (fausses connexions, faux `@champion`, fausse `train_and_promote`). Les variables `POSTGRES_*` y sont retirées pendant chaque test ;
- `tests/db` : l'application complète contre un vrai PostgreSQL, configuré par les variables `POSTGRES_*` (`src/db/connection.py`).

Sans base, les 24 tests de `tests/db` sont sautés :

```bash
uv run pytest tests/api tests/db
```

Résultat attendu : 511 tests réussis et 24 sautés, avec un avertissement final qui rappelle que les tests PostgreSQL n'ont pas tourné.

Avec le PostgreSQL de Compose (depuis la racine du dépôt, `.env.production` rempli) :

```bash
docker compose up -d --wait postgres
POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=5433 uv run --env-file .env.production pytest tests/api tests/db
```

Sous PowerShell :

```powershell
docker compose up -d --wait postgres
$env:POSTGRES_HOST = "127.0.0.1"; $env:POSTGRES_PORT = "5433"
uv run --env-file .env.production pytest tests/api tests/db
```

Résultat attendu : 535 tests réussis, aucun sauté.

Pour séparer tout à fait les tests de l'historique, on peut aussi les lancer sur une base dédiée du même serveur, créée une fois : `docker compose exec postgres createdb -U neuraxion neuraxion_test` (l'utilisateur est celui de `.env.production`), puis ajouter `POSTGRES_DB=neuraxion_test` devant la commande ci-dessus. `uv run --env-file` lit l'utilisateur, le mot de passe et la base dans `.env.production` ; les variables déjà posées dans le terminal passent avant celles du fichier, si bien que `POSTGRES_HOST=postgres` et `POSTGRES_PORT=5432`, valables entre conteneurs, sont remplacés par l'adresse vue depuis la machine. Écrire `127.0.0.1` plutôt que `localhost` : sous Windows, `localhost` essaie d'abord `::1`, où Compose ne publie pas le port, et chaque connexion perd alors 2 s (`GET /health` répond `degraded`).

Règles de `tests/db/conftest.py` :

- sans `POSTGRES_USER`, `POSTGRES_PASSWORD` ou `POSTGRES_DB`, chaque test est sauté et le motif le dit ;
- `POSTGRES_PORT` qui n'est pas un entier, ou base configurée mais injoignable : chaque test échoue, avec le type de l'erreur et l'adresse à vérifier, pour qu'une base arrêtée ne passe pas pour un succès ;
- chaque test travaille dans un schéma temporaire `neuraxion_test_...`, supprimé à la fin. `PGOPTIONS` le place seul dans le `search_path` de toutes les connexions du processus : les connexions ouvertes par l'API et par les tests n'écrivent pas dans la table `inferences` du schéma `public`, qui garde l'historique de la base de Compose (un test le vérifie). Un code qui changerait lui-même son `search_path` échapperait à cette protection : `src/` n'en contient aucun. `PGSERVICE` et `PGSERVICEFILE` sont retirés, car un fichier de service passerait avant `PGOPTIONS`. Avant chaque test, la fixture vérifie que `current_schema()` est bien le schéma temporaire, et sinon échoue sans lancer le test ;
- un processus tué en plein test (Ctrl+C répété, coupure) peut laisser un schéma `neuraxion_test_...`. Pour les lister, quand aucun test ne tourne : `SELECT nspname FROM pg_namespace WHERE nspname LIKE 'neuraxion\_test\_%';`, puis `DROP SCHEMA <nom> CASCADE;` pour chacun ;
- un test crée un rôle temporaire `neuraxion_api_limitee_...`, qui a seulement le droit d'insérer, puis le supprime. L'utilisateur de `.env.production` doit donc pouvoir créer un schéma et un rôle, ce qui est le cas de l'utilisateur créé par l'image `postgres` ;
- le `@champion` (R1) est remplacé par un faux module `src.registry.champion`, et `train_and_promote` (R2) par une fausse fonction : ni MLflow ni entraînement réel.

Vérifications du §13 et de l'issue #22 :

| Vérification | Sans base (`tests/api`) | Avec PostgreSQL (`tests/db`) |
|---|---|---|
| §13 n° 3 : `POST /predict` avec une entrée valide et une entrée invalide | `test_predict.py` (`test_prediction_returns_a_decision_and_records_one_row_with_the_version`, `test_unknown_fields_are_refused_and_not_recorded`, `test_validation_errors_are_not_recorded`) | `test_api_postgres.py::test_valid_then_invalid_request_and_the_documented_query` |
| §13 n° 4 et critère I : une prédiction réussie est enregistrée avec la version du modèle | `test_predict.py` (même test, sur une fausse connexion ; `test_model_name_defaults_to_the_environment_then_credit_default` pour le nom) | `test_predict_postgres.py::test_prediction_creates_a_row_with_the_version` et `test_the_row_records_the_name_of_the_served_model` ; `test_api_postgres.py` : requête SQL de la section « Table inferences », version après une promotion, lignes gardées après un redémarrage |
| `POST /train` sans clé et avec une mauvaise clé | `test_train.py` (`test_without_header_answers_401_in_the_common_format`, `test_a_wrong_key_answers_403`) | `test_api_postgres.py::test_train_without_key_or_with_a_wrong_key_changes_nothing` : 401 ou 403, ni entraînement, ni rechargement, ni ligne en base ; `test_train_stays_closed_when_the_server_key_is_not_configured` : 403 si `TRAIN_API_KEY` est absente, vide ou faite d'espaces |

## Points à confirmer à la revue des contrats (C6)

- Le format de `contracts/features.yaml` (lot P1). `build_request_model()` suppose pour chaque variable : `name` et `type` (`numeric`, `integer` ou `categorical`) obligatoires ; `min`, `max`, `allowed`, `required` (vrai par défaut), `label`, `unit` et `example` facultatifs.
- Le format de `contracts/examples.json` (lot P1) : un objet profil → demande complète, par exemple `{"prudent": {...}, "moyen": {...}, "risque": {...}}`.
- Les types stricts de la requête : l'interface envoie des nombres JSON, jamais du texte.
- Les valeurs `agree` et `disagree` du feedback.
- Le raccordement A1 → A2 : A1 construit la classe de requête et appelle `predict.build_router()` ; A2 remplit le corps de la route sans changer sa signature.
- Les choix propres à C4 : codes d'erreur supplémentaires, `GET /health` toujours en 200 avec `ok` ou `degraded`.

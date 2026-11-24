# Neuraxion

Neuraxion estime la probabilité qu'un prêt LendingClub de 36 mois ne soit pas remboursé, puis propose de l'accorder ou de le refuser. Le dépôt contient à la fois l'exploration (notebooks Jupyter suivis dans MLflow) et l'application de production : une API FastAPI, une interface React, un serveur MLflow et une base PostgreSQL, lancés ensemble avec Docker Compose.

Ce README est encore un squelette. Les sections marquées (à compléter) seront remplies au fil du projet.

## Prérequis

Pour lancer l'application, il faut Docker avec Docker Compose, au moins 4 Go de mémoire et 3 Go de disque libres, et un accès à Internet au premier démarrage. Pour travailler sur le code et les notebooks, il faut aussi [uv](https://docs.astral.sh/uv/). Inutile d'installer Python 3.12 à l'avance : uv le télécharge s'il manque.

## Configuration

Les variables de production sont décrites dans `.env.example`. On copie ce fichier, puis on renseigne les valeurs vides, au minimum `POSTGRES_PASSWORD` et `TRAIN_API_KEY` :

```bash
cp .env.example .env.production
```

Les notebooks ont leur propre fichier :

```bash
cp .env.experiment.example .env.experiment
```

Ni `.env.production` ni `.env.experiment` ne sont versionnés.

## Données

(à compléter)

Les données viennent du jeu Kaggle Lending Club (fichier `accepted_2007_to_2018Q4.csv.gz`). Elles sont téléchargées dans `data/raw/`, un dossier exclu de Git. Le jeton Kaggle (`KAGGLE_API_TOKEN`) est facultatif : il ne sert que si Kaggle refuse le téléchargement anonyme.

## Lancement avec Docker Compose

(à compléter)

Pour l'instant, seul PostgreSQL est déclaré :

```bash
docker compose up postgres
```

Depuis la machine, la base répond sur `localhost:5433`. Ce port évite le conflit avec un PostgreSQL déjà installé sur le port habituel, 5432.

## Travail en local

On installe l'environnement, puis on lance MLflow local dans un premier terminal et Jupyter dans un second :

```bash
uv sync
uv run mlflow server --host 127.0.0.1 --port 5001
uv run --env-file .env.experiment jupyter lab
```

MLflow local utilise le port 5001, car le port par défaut (5000) est pris par AirPlay sur macOS. Le code de `src/` s'importe directement depuis les notebooks, par exemple avec `from src.training import ...`.

## Tests et vérifications

```bash
uv run pytest
```

Le dépôt ne doit contenir aucun fichier de données. Cette commande ne doit rien afficher :

```bash
git ls-files | grep -Ei '\.(csv|parquet)(\.gz)?$'
```

## Démonstration

(à compléter)

## Structure du dépôt

Certains dossiers seront créés au fil des tâches.

```text
notebooks/            explorations, comparaisons et conclusions
src/data/             téléchargement, ingestion et lecture des données
src/training/         cleaning, découpage, pipeline et validation
src/registry/         MLflow : suivi des runs, promotion et chargement du modèle actif
src/api/              application FastAPI et schémas
src/db/               accès à PostgreSQL et création du schéma
src/ui/               interface React
tests/                un dossier par lot : data, training, registry, api, db
contracts/            contrat de données et exemples
config/               protocole d'entraînement (training.yaml)
scripts/              démarrage du backend et remise à zéro
docs/                 documentation et schéma d'architecture
data/                 données locales, exclues de Git
docker-compose.yml    orchestration des services
pyproject.toml        dépendances, avec uv.lock pour les versions exactes
.env.example          variables de production, sans secret
```

## Organisation de l'équipe

| Lot | Rôle |
|---|---|
| P1 | Données et exploration |
| P2 | Pipeline ML |
| P3 | MLflow |
| P4 | API et base |
| P5 | Interface et intégration |

La branche `main` contient l'état livrable, avec un tag par jalon. La branche `dev` sert à l'intégration et reste la branche par défaut pendant le projet. Pour récupérer la version livrée, il faut donc cloner avec `git clone -b main`.

Chaque issue a sa branche (`feat/NN-sujet`, `fix/NN-sujet` ou `docs/NN-sujet`) et sa pull request vers `dev`, avec `Closes #NN` dans la description. Le relecteur du lot approuve, puis la PR est fusionnée par squash. Les messages de commit suivent le format `type(portée): description (#NN)` et sont écrits en français.

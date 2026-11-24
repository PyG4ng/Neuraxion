# Protocole de validation

Ce document décrit comment on entraîne et comment on juge les modèles du projet. Il vaut pour POST /train comme pour les notebooks de comparaison, qui suivent tous les mêmes règles. Les valeurs sont dans `config/training.yaml`, que le code lit avec `load_config()`. Pour changer une valeur, on la change dans ce fichier et nulle part ailleurs (P2 le tient à jour). Le code se trouve dans `src/training/`.

## Périmètre et cible

On garde les prêts de 36 mois accordés entre juin 2007 et décembre 2015, à condition que leur statut soit terminé. Un prêt encore en cours n'a pas d'issue connue : le compter comme remboursé ou comme perdu fausserait la cible. Les prêts de 60 mois sont laissés de côté pour une raison proche. Un prêt de 60 mois accordé en 2015 se termine en 2020, alors que le jeu de données s'arrête fin 2018, donc une bonne partie de ces prêts n'est pas terminée. Le modèle ne couvre que les prêts de 36 mois, et la durée n'est pas un champ du formulaire.

La cible vaut 1 quand le prêt n'a pas été remboursé et 0 quand il l'a été. `clean()` la construit à partir de la colonne `loan_status`, dans une colonne appelée `target` :

| Statut dans le jeu de données | Cible |
|---|---|
| Charged Off | 1 |
| Default | 1 |
| Does not meet the credit policy. Status:Charged Off | 1 |
| Fully Paid | 0 |
| Does not meet the credit policy. Status:Fully Paid | 0 |

Les autres statuts (Current, In Grace Period et les différents retards Late) désignent des prêts en cours et sont écartés. Le contrat de données de P1 peut rappeler cette définition, mais c'est `config/training.yaml` que le code lit.

## Découpage dans le temps

On découpe selon la date d'octroi (`issue_d`) et non au hasard. Le modèle servira à juger des demandes futures, donc on le valide sur des prêts plus récents que ceux qui ont servi à l'entraîner. Un découpage au hasard mélangerait les années. Comme le taux de défaut et le profil des emprunteurs changent d'une année à l'autre, il donnerait un score trop optimiste.

| Partie | Période, bornes incluses | Rôle | Où sont les données |
|---|---|---|---|
| Entraînement | juin 2007 à décembre 2013 | ajuster le modèle | table d'entraînement PostgreSQL, `dev.parquet` |
| Validation | janvier à décembre 2014 | promotion, seuil de décision, comparaisons | table d'entraînement PostgreSQL, `dev.parquet` |
| Test | janvier à décembre 2015 | une seule mesure finale | `test.parquet` uniquement |

La validation est l'année qui précède le test. C'est ce qui ressemble le plus à la situation du test quand les données dérivent. Le modèle servi est celui qui a été entraîné sur 2007-2013. On ne le réentraîne pas sur 2007-2014, sinon la métrique qui a décidé de sa promotion ne serait plus celle du modèle en service.

La graine 42 sert partout : pour le tirage du préréglage rapide, pour les modèles qui ont une part de hasard et dans les notebooks. Deux exécutions sur les mêmes données donnent donc le même résultat.

## Pourquoi POST /train ne lit jamais le test

Le test sert à estimer une seule fois la performance du modèle final, sur des prêts qu'il n'a jamais vus. Si on s'en servait pour choisir quoi que ce soit (un modèle, une variable, un seuil), il deviendrait une seconde validation. La mesure finale serait alors trop optimiste, parce que nos choix auraient été faits pour bien marcher sur ces prêts-là. Or POST /train peut être lancé autant de fois qu'on veut, avec différents préréglages. S'il avait accès à 2015, chaque appel serait une occasion de regarder le test.

Trois verrous l'en empêchent. Le premier tient à la source des données : POST /train ne lit que la table d'entraînement de PostgreSQL, et l'ingestion n'y charge que les prêts de 2007 à 2014. Les prêts de 2015 n'entrent jamais dans la base. Le deuxième est dans le code : `temporal_split()` lève une erreur dès qu'une ligne est datée du 1er janvier 2015 ou après, et un test de la tâche M1 vérifie que l'entraînement s'arrête avant cette date. Le troisième porte sur l'accès au fichier : `test.parquet` n'est lu que par `read_test_frame()`, et seul le notebook 08 l'appelle, une fois, après le choix final (tâche D5). L'exploration ne regarde de 2015 que le nombre de prêts et la part de prêts terminés.

## Métrique et seuil de promotion

La métrique de promotion est la ROC-AUC calculée sur la validation 2014. On peut la lire comme la probabilité qu'un prêt en défaut tiré au hasard reçoive un score de risque plus élevé qu'un prêt remboursé tiré au hasard. Elle vaut 0,5 pour un modèle qui répond au hasard et 1 pour un classement parfait. Elle juge le classement des prêts sans dépendre du seuil de décision, et elle se compare directement au grade de LendingClub.

Un modèle est promu si `metric_new >= threshold`, comme le demande le §6 du cahier des charges. Le seuil est la ROC-AUC du grade seul sur 2014 (A=1 à G=7), calculée sans entraînement à partir des données ingérées. L'idée est qu'un modèle qui n'utilise ni le grade ni le taux d'intérêt doit faire au moins aussi bien que la note de LendingClub. P1 mesure cette valeur dans le notebook 01 (tâche D2) et P2 l'inscrit dans `promotion.threshold` (tâche M2). En attendant, le fichier contient 0,65, une valeur provisoire signalée en commentaire. Le seuil est lu dans la configuration et n'est jamais écrit en dur dans le code.

## Seuil de décision

Le seuil de décision est la probabilité de défaut à partir de laquelle on refuse un prêt. Il ne faut pas le confondre avec le seuil de promotion, qui porte sur la ROC-AUC. La règle est la même partout, dans `evaluate()` comme dans /predict : on refuse si la probabilité de défaut est supérieure ou égale au seuil, on accepte sinon.

POST /train choisit ce seuil sur la validation 2014 avec `choose_decision_threshold()`. La fonction essaie les seuils de 0,01 à 0,99 par pas de 0,01. Pour chacun, elle additionne le profit réalisé des prêts acceptés, c'est-à-dire `total_pymnt` moins `funded_amnt`, et elle garde le seuil qui donne la plus grande somme (le plus petit en cas d'égalité). Le seuil retenu est enregistré avec le run et sur la version du modèle (tag `decision_threshold`, voir les conventions MLflow de P3). /predict le renvoie dans sa réponse, et chaque ligne de la table des inférences le conserve. Dans le code comme dans l'interface, les deux noms restent distincts : `promotion_threshold` et `decision_threshold`.

## Métriques suivies

`evaluate()` renvoie toujours les mêmes sept clés, calculées sur la validation. Seule `roc_auc` décide de la promotion. Les autres servent à comparer les modèles et à expliquer les choix.

| Clé | Ce qu'elle mesure |
|---|---|
| `roc_auc` | qualité du classement des prêts, du plus sûr au plus risqué |
| `pr_auc` | aire sous la courbe précision-rappel, plus parlante quand les défauts sont rares |
| `brier` | écart moyen au carré entre la probabilité prédite et l'issue réelle (plus il est bas, mieux c'est) |
| `precision` | part de vrais défauts parmi les prêts refusés |
| `recall` | part des défauts qui sont bien refusés |
| `acceptance_rate` | part des prêts acceptés |
| `realized_profit` | somme de `total_pymnt` moins `funded_amnt` sur les prêts acceptés, en dollars |

## Préréglages de POST /train

POST /train reçoit `{"preset": "default" | "rapide" | "faible"}`. Les trois préréglages gardent la même validation (toute l'année 2014) et la même graine.

| Préréglage | Données d'entraînement | Modèle | Durée visée | Usage |
|---|---|---|---|---|
| default | toute la période 2007-2013 | configuration de référence (régression logistique, T1, V1), remplacée par le choix final en M4 | moins de 2 minutes | premier entraînement au démarrage |
| rapide | 50 000 prêts tirés avec la graine | régression logistique, T1, V1 | moins de 30 secondes | démonstration en direct |
| faible | toute la période 2007-2013 | régression logistique sur `loan_amnt` et `annual_inc` | pas d'objectif | montrer un refus de promotion |

Les deux variables du préréglage faible doivent exister dans le contrat de données. On s'attend à ce qu'elles prédisent mal le défaut à elles seules, ce qui doit placer le modèle nettement sous le seuil. La première exécution réelle (tâche M2) le vérifiera.

## Comparaisons dans les notebooks

Les notebooks 03 à 06 comparent des choix autour d'une configuration de référence : régression logistique, transformations T1 et variables V1 (section `reference` du fichier). Chaque notebook ne change qu'un facteur. Les résultats restent ainsi comparables d'un notebook à l'autre, et la synthèse (notebook 07) peut ensuite combiner les meilleurs choix. Tous utilisent le même découpage, la même fonction `evaluate()` et la même graine. La section `reference` ne bouge pas quand M4 met à jour default : relancés avant la soutenance, les notebooks donnent donc les mêmes résultats.

V1 regroupe les variables du contrat de données. V2 ajoute à V1 quatre variables construites : montant sur revenu, solde renouvelable sur revenu, ancienneté du crédit et FICO moyen. T1 remplace les valeurs manquantes par la médiane (ou par la modalité la plus fréquente), standardise les variables numériques et code les catégories en one-hot. T2 ajoute des indicateurs de valeurs manquantes, passe les montants au log1p et regroupe les modalités rares.

Pour changer un facteur, on passe à `run_pipeline()` un dictionnaire qui ne contient que ce qui change. Il est appliqué par-dessus la référence :

```python
from src.data import read_dev_frame
from src.training import comparison_tags, run_pipeline

df = read_dev_frame()  # 2007 à 2014, jamais 2015
result = run_pipeline(df, {"model": "hist_gradient_boosting"})
tags = comparison_tags(axe="modele", variante="hgb", lot="P2")
# {'axe': 'modele', 'variante': 'hgb', 'lot': 'P2', 'protocole': 'v1'}
```

Chaque run de comparaison est enregistré dans l'expérience MLflow `credit-exploration` avec quatre tags : `axe`, `variante`, `lot` et `protocole`. `comparison_tags()` les construit et refuse les valeurs inconnues, car une faute de frappe dans un tag ferait disparaître le run du tableau de synthèse.

| axe | Ce qui est comparé | Lot | Notebook |
|---|---|---|---|
| reference | grade seul, qui fixe le seuil de promotion | P1 | 01 |
| modele | régression logistique, HistGradientBoosting, forêt aléatoire si le temps le permet | P2 | 03 |
| pipeline | V1 contre V2 | P3 | 04 |
| transformation | T1 contre T2 | P4 | 05 |
| seuil | profit et taux d'acceptation selon le seuil, face à la politique qui refuse les grades E, F et G | P5 | 06 |
| synthese | combinaison des meilleurs choix | P2 | 07 |
| test_final | modèle retenu évalué sur 2015, une seule fois | P1 | 08 |

`variante` est un nom court et libre (logreg, hgb, V2, T2...). `protocole` vaut `v1`. Si le protocole change (dates, cible ou métrique), on passe à `v2` pour ne pas mélanger des runs qui ne se comparent plus.

## Fonctions de src/training

| Fonction | Fichier | Rôle | Écrite par |
|---|---|---|---|
| `load_config()`, `load_contract()`, `resolve_preset()`, `comparison_tags()` | config.py | lire le protocole et le contrat, compléter un préréglage, construire les tags | C2, déjà réelles |
| `clean(df)` | cleaning.py | cible, valeurs impossibles, doublons, variables dérivées, types | M1 |
| `temporal_split(df, cfg)` | split.py | entraînement et validation, refus de 2015 | M1 |
| `build_pipeline(preset, contract)` | pipeline.py | pipeline scikit-learn non entraîné : variables, transformations, modèle | M2 |
| `evaluate(y, proba, decision_threshold, profit=None)` | validation.py | les sept métriques | M2 |
| `choose_decision_threshold(y, proba, profit)` | validation.py | seuil qui maximise le profit réalisé | M2 |
| `run_pipeline(df=None, preset="default")` | pipeline.py | tout l'enchaînement, sans MLflow | M2 |

`run_pipeline()` n'appelle jamais MLflow. Elle renvoie un `PipelineResult` : modèle entraîné, paramètres, métriques, nom et valeur de la métrique de promotion, seuil de décision, colonnes d'entrée, exemple d'entrée et version du contrat. P3 l'enregistre dans MLflow et décide de la promotion avec `should_promote()`. Sans DataFrame, `run_pipeline()` lit elle-même la table d'entraînement de PostgreSQL avec `read_training_frame()`, et c'est ainsi que `train_and_promote(preset)` l'appelle. Les notebooks lui passent `read_dev_frame()`, et les tests des données factices.

Toute la transformation des variables se trouve dans le pipeline scikit-learn que renvoie `build_pipeline()`. L'entraînement et /predict appliquent donc exactement les mêmes règles, comme le demande le §6 du cahier des charges.

Jusqu'à la fusion de M1 et M2, ces fonctions ont un corps factice qui renvoie des valeurs et jamais d'erreur. Il sert de bouchon aux autres lots. `run_pipeline()` entraîne un modèle qui prédit toujours le taux de défaut moyen et annonce une ROC-AUC fixe. Elle vaut 0,90 pour default, rapide et les dictionnaires, ce qui donne une promotion acceptée avec le seuil provisoire, et 0,50 pour faible, ce qui donne un refus. `evaluate()` renvoie les sept clés avec des valeurs fixes et `choose_decision_threshold()` renvoie 0,5. Chaque appel à un corps factice écrit un avertissement dans les logs, pour qu'on ne prenne pas ces chiffres pour de vrais résultats.

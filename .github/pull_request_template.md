Closes #

## Ce qui change

<!-- Deux ou trois phrases : ce que fait la PR et pourquoi. -->

## Comment vérifier

<!-- Les commandes à lancer et le résultat attendu. -->

## Avant de demander la relecture

- [ ] Le titre suit le format `type(portée): description (#NN)`
- [ ] `uv run pytest` passe sur le dossier de tests du lot
- [ ] Les notebooks modifiés ont été relancés de zéro, sorties conservées
- [ ] `git ls-files | grep -Ei '\.(csv|parquet)(\.gz)?$'` n'affiche rien
- [ ] Aucun secret ni fichier `.env` réel n'est ajouté
- [ ] Les commandes et résultats utiles au README sont dans cette description (P5 les reprend)

# Projet — Détecter la transaction d'un client

**Équipe de 3 · 6 semaines · rendu par dépôt**

---

## Le contexte

Une banque observe ses clients et veut savoir **lesquels réaliseront une transaction** dans une
fenêtre donnée : pour cibler une offre, et pour dimensionner son dispositif.

## La question à laquelle le projet répond

> Peut-on repérer, dans les 200 variables d'un client, ceux qui vont réaliser une transaction — et de
> combien une méthode soignée bat-elle le tri que la banque applique aujourd'hui sans modèle ?

## Ce que vous recevez

| Fichier | Contenu |
|---|---|
| `train.csv` | **160 000 clients** dont l'issue est connue — 202 colonnes, dont `target` |
| `a_decider.csv` | **40 000 clients** — les 201 mêmes colonnes, **sans `target`** |

**10,05 % des clients de `train.csv` ont réalisé la transaction** (20 098 sur 200 000 au total).

**Trois avertissements :**

- **les 200 variables sont continues et anonymisées** : `var_0` … `var_199`, sans aucun nom parlant.
  Il n'y a pas de sens métier à retrouver, il y a des régularités à mesurer ;
- **aucune valeur ne manque** et il n'y a aucune variable catégorielle : votre travail se joue donc
  sur **ce que vous faites de ces nombres**, et sur **lesquels vous gardez** ;
- **la métrique d'évaluation est l'aire sous la courbe ROC (AUC)**.

Aucune donnée ne permet d'identifier une personne : `ID_code` est un numéro interne.


## Ce que contient chaque colonne

| Colonne | Type | Ce que c'est |
|---|---|---|
| `ID_code` | texte | identifiant du client |
| `var_0` … `var_199` | **décimal** | **200 variables continues anonymisées** |
| `target` | 0/1 | **la cible** : 1 = transaction réalisée |

**Ce qui est mesuré sur les données livrées, et qui doit orienter votre travail :**

- les 200 variables s'étendent de **−73,8 à +65,4** : ce sont des valeurs **centrées**, pas des
  montants, et **124 d'entre elles prennent des valeurs négatives** ;
- **aucune valeur manquante** ;
- **aucun nom parlant** : il n'existe aucune description publiée de `var_57`. Vous ne pouvez pas
  raisonner par le métier — vous devez **mesurer** ce que chaque variable apporte, seule puis à
  plusieurs.


**Ce que Kaggle documente, et ce qu'il ne documente pas.** La page *Data* de cette compétition ne
publie **aucune description par colonne** : les variables sont anonymisées et la compétition ne donne
que leurs familles. **Les lignes ci-dessus sont donc tout ce qui est documenté** — vous ne trouverez
pas, en ligne, à quoi correspond une variable précise. C'est une contrainte du sujet, et elle est
volontaire : votre travail est de **mesurer** ce que chaque variable apporte, pas de l'interpréter.

## Ce que vous rendez

**Un dépôt de code**, pas un notebook :

1. un **pipeline exécutable** qui prend `a_decider.csv` et produit `ID_code,target` — une **décision** ;
2. le **protocole de validation** retenu, justifié ;
3. les **essais**, y compris ceux qui ont échoué, avec leur score ;
4. une **note de 3 pages** : décision, métrique et seuil, méthode, résultat chiffré, ce qui a échoué ;
5. une **model card** ;
6. la **fiche de données** du S13.

Le pipeline doit tourner **en moins de 10 minutes sur CPU**, sans GPU, avec `scikit-learn` seul.

## Ce que l'enseignant garde — et ce que vous pouvez demander

**Vous recevez toutes les lignes, sauf mention contraire ci-dessous.** Ce qui reste chez l'enseignant
n'est **jamais** une partie des données : c'est **la cible** des lignes que vous devez estimer, et elle
seule. Sans elle, il n'y aurait rien à prédire.

| | |
|---|---|
| Fichier retenu | **`verite.csv`** |
| Lignes | **40 000** |
| Colonnes | `ID_code`, `target` |
| Contenu | l'issue réelle des clients de `a_decider.csv` |

**Vous n'y avez pas accès, et aucune copie ne doit circuler.** C'est l'enseignant qui mesure votre
rendu contre ce fichier.

### La table source complète est disponible sur demande

Les données de ce sujet viennent d'une **compétition publique**. Si vous voulez travailler sur la
**table d'origine entière** — plus de lignes, plus de variables, ou les fichiers annexes — **demandez-la
à l'enseignant** : il l'a, et il peut vous la fournir pour un usage d'étude. Précisez dans votre
journal ce que vous en avez fait, et **restez sur la même découpe** : le rendu final porte sur les
lignes de `a_estimer.csv` / `a_decider.csv` / `horizon.csv`, pas sur la table complète.

## Travailler : récupérer les données, coder, sauvegarder

Les données du cours sont sur le **Drive du cours**, en lecture seule — **dossier du sujet :**
**https://drive.google.com/drive/folders/1ftIdcRddFC_exiwx7i_yBKjEuXtnHdYz?usp=sharing**. Vous téléchargez **le dossier de votre sujet** — pas les autres — et vous choisissez
ensuite **où coder**. Le lien est rappelé au **premier écran du carnet de démarrage**.

Le dossier contient **trois choses** : cet énoncé, le **carnet de démarrage** (`Carnet_de_demarrage.ipynb`)
et les fichiers de données.

### 1. Récupérer votre dossier

| | |
|---|---|
| Dossier du sujet | **`Sujet_07_Santander_Transaction/`** |
| Poids attendu | **302,3 Mo** |
| Contenu | les fichiers listés dans « Ce que vous recevez » |

Après téléchargement, vérifiez que rien n'est tronqué : le poids du dossier doit correspondre, et
chaque `csv` doit s'ouvrir avec le bon nombre de lignes.

### 2. Option A — travailler en local, dans VS Code

C'est la voie la plus simple : vos fichiers sont sur votre disque, rien ne disparaît.

```
python -m venv .venv
```

```
source .venv/bin/activate
```

*(sous Windows : `.venv\Scripts\activate`)*

```
pip install pandas scikit-learn
```

Ouvrez le dossier dans VS Code, écrivez votre pipeline — `pipeline.py` par exemple — et lancez-le :

```
python pipeline.py
```

**La contrainte des 10 minutes s'applique à votre machine.** Mesurez-la : elle fait partie du
livrable, et un pipeline qui tourne en 40 minutes sur votre portable ne sera pas accepté.

### 3. Option B — travailler dans un carnet Kaggle

Kaggle ne voit pas votre disque : il faut **lui donner vos fichiers**, en créant votre propre jeu de
données. C'est vous qui le déposez, pour votre usage — le jeu est **privé par défaut**.

1. **kaggle.com/datasets** → **New Dataset**.
2. Déposez les fichiers du dossier `Sujet_07_Santander_Transaction/` (glisser-déposer), nommez le jeu de données
   `esgc-vak-sujet-07-…`, puis **Create**.
3. Dans votre carnet : **Add Input** → onglet **Datasets** → filtre **Your Datasets** → sélectionnez
   le vôtre → **Add**.
4. Les fichiers apparaissent dans **`/kaggle/input/<nom-de-votre-jeu>/`**, en **lecture seule**.
   Vérifiez avant d'écrire du code :

```
!ls /kaggle/input/<nom-de-votre-jeu>/
```

5. Réglez la session : **Session options** → **Accelerator : None** — le projet tient sur CPU, c'est
   la contrainte du cours. Laissez **Internet désactivé**.

> **La compétition d'origine n'est pas la source.** Ne cherchez pas à l'ajouter : ses fichiers
> diffèrent des vôtres — pas d'horizon sans cible, pas de jeu à estimer — et plusieurs de ces
> compétitions ne sont plus rejoignables.

### 4. Sauvegarder, et réutiliser votre travail

**En local (Option A).** Votre dossier de travail **est** la sauvegarde. Gardez les sorties dans un
sous-dossier `sorties/`, et versionnez avec `git` si vous le souhaitez.

**Sur Kaggle (Option B).** `/kaggle/input` est en **lecture seule** : écrivez dans
**`/kaggle/working/`**.

```python
import os
os.makedirs("/kaggle/working/sorties", exist_ok=True)
prediction.to_csv("/kaggle/working/sorties/prediction.csv", index=False)
```

- **Sauvegarder** : bouton **Save Version** → **Save & Run All (Commit)**. Le carnet est **ré-exécuté
  de bout en bout** dans un conteneur propre, et **tout ce qui reste dans `/kaggle/working/` est
  conservé** comme sortie de cette version.
- **Reprendre ces sorties dans un autre carnet** : **Add Input** → onglet **Notebook Output** →
  votre carnet, la version voulue → le contenu apparaît dans `/kaggle/input/<nom-du-carnet>/`.
- **⚠️ `/kaggle/working` est éphémère** : une session inactive est coupée au bout de 20 minutes. Sans
  **Save Version**, tout est perdu.

**Quel que soit votre choix**, le livrable est le même : un **dépôt de code** reproductible, dont le
pipeline tourne en moins de 10 minutes sur CPU.

## Les trois temps du projet

### Bloc 1 — avant la séance 5 · *la décision et la mesure*

- **La métrique** : ce qu'elle récompense, ce qu'elle ignore.
- **Le coût** : une vente manquée contre une sollicitation inutile. Chiffrez les deux.
- **La règle métier** : comment la banque trie-t-elle aujourd'hui, sans modèle ? Mesurez-la. C'est
  votre plancher.
- **Le protocole de validation** : comment vous découpez `train.csv`, et pourquoi.

### Bloc 2 — avant la séance 9 · *le pipeline*

- Le pipeline complet, exécutable d'un bout à l'autre.
- **Au moins huit essais**, dont ceux qui ont échoué, chacun avec son score.
- **La question de ce sujet** : que vaut **chaque variable prise seule** ? Et que vaut le modèle
  quand on n'en garde qu'une, cinq, dix, cinquante, cent ? **Tracez cette courbe.** C'est elle qui
  décidera de votre sélection — et elle vous apprendra quelque chose sur la table que les autres
  sujets ne vous apprendraient pas.
- **L'analyse d'erreurs** : sur quels clients vous vous trompez.

### Bloc 3 — avant la séance 13 · *le dépôt et la robustesse*

- Le dépôt propre, la note de 3 pages, la model card, la fiche de données.
- **Trois mutations nommées**, chacune applicable en **une cellule**.
- Le test de robustesse, et la décision finale sur les 40 000 clients.

## Comment vous êtes évalués

Voir la grille du cours. Trois points pèsent particulièrement ici :

- **la justification de la sélection de variables** : garder les 200 « au cas où » n'est pas un choix,
  c'est une absence de choix — et elle se mesure, en temps comme en qualité ;
- **le chiffre contre le plancher** : votre score se rapporte à une règle sans modèle, mesurée ;
- **la vérité du journal** : le journal consigne les essais, y compris ceux qui ont échoué.

## Deux règles du cours

**Pas de fuite.** Une information inconnue **au moment où la décision est prise** ne peut pas entrer
dans le modèle.

**Le code doit rester réparable.** On doit pouvoir passer d'une version à l'autre en changeant une
cellule, pas en réécrivant le pipeline.

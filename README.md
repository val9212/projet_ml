# Prédiction de fins de phrases musicales

## OBJECTIFS : 
Une séquence musicale peut être jouée ou écrite, comme on peut énoncer un texte ou l'écrire. Comme un texte, une séquence musicale est
Structurée. Cette structure peut se décrire comme une suite de phrases. De ce fait, il est possible de déterminer toutes les fins 
De phrases à l'aide de nombreux facteurs. Évidemment, nous ne réaliserons pas ce travail à la main en écoutant chaque morceau, nous allons utiliser des corpus annotés. En les utilisant, nous pouvons mettre en place des algorithmes d'apprentissage capables de prédire les fins de phrases.

Ainsi, l'objectif de ce projet est de préparer des données, les explorer, construire et comparer des modèles prédictifs de fin de phrase musicale.

## MATERIELS : 

### Jeu de données

Pour répondre aux objectifs de ce projet, nous avons récupéré des données issues du Meertens Tune Collections 1. C’est
un corpus de mélodies des Pays-Bas annotés par un ensemble d'attributs : 
Un objet mélodique contient des champs de métadonnées et plusieurs séquences de valeurs de caractéristiques.


Les champs de métadonnées sont : 

- **id** `(string)`: Identifiant unique de la mélodie.
- **type** `({vocal, instrumental})`: Indique le type de chanson (vocal / instrumental).
- **year** `(int)`: Année de publication de la chanson.
- **tunefamily** `(string)`: Identifiant de la famille mélodique.
- **tunefamily_full** `(string)`: Nom complet de la famille mélodique.
- **freemeter** `(bool)`: Indique si la mélodie a une structure rythmique (True / False).
- **ann_bgcorpus** `(bool)`: Spécifie si la mélodie est indépendante des chansons d'un autre corpus, en l'occurrence MTC-ANN-2.0.1 (True / False).
- **origin** `(string)`: Indique le chemin du fichier kern dans la collection ESSEN. Ce chemin renseigne aussi sur l'origine géographique de la mélodie.

Les séquences caractéristiques correspondent aux séquences de notes dans une mélodie donnée. Elles renseignent sur les 61 caractéristiques distinctes: 

- **pitch** `(string)` : Représentation de la hauteur de la note selon le format de music21.
- **midipitch** `(int)` : Numéro MIDI représentant la hauteur de la note (de 0 à 108).
- **pitch40** `(int)` : Représentation de la hauteur en base 40.
- **contour3** `(string)` : Contour de la hauteur par rapport à la note précédente ('-' : Descendante, '=' : Égale, '+' : Montante, la première note a une valeur None).
- **contour5** `(string)` : Contour détaillé de la hauteur par rapport à la note précédente ('--' : Descente d’au moins 3 intervalles MIDI, '-' : Descente normale, '=' : Égale, '+' : Ascension normale, '++' : Ascension d’au moins 3 intervalles MIDI, la première note a une valeur None).
- **diatonicinterval** `(int)` : Intervalle diatonique entre la note actuelle et la précédente (en degrés d’échelle, la première note a une valeur None).
- **chromaticinterval** `(int)` : Intervalle chromatique (différence midipitch) entre la note actuelle et la précédente (La première note a une valeur None).
- **tonic** `(string)` : Classe de hauteur de la tonique pour la note actuelle (A à G avec altérations).
- **mode** `(string)` : Mode de la note.
- **scaledegree** `(int)` : Degré de la hauteur par rapport à l’échelle de la tonique (de 1 à 7).
- **scaledegreespecifier** `(string)` : Spécification du degré.
- **diatonicpitch** `(int)` : Hauteur diatonique de la note.
- **timesignature** `(Fraction)` : Signature rythmique de la note, sous forme d’une fraction (Si aucune signature n’est notée, la valeur est None).
- **beatstrength** `(float)` : Force métrique (entre 0.0 et 1.0) du temps d’apparition de la note (music21).
- **metriccontour** `(string)` : Contour de la force métrique par rapport à la note précédente ('-' : Plus faible, '=' : Égale, '+' : Plus forte).
- **imaweight** `(float)` : Poids métrique (entre 0.0 et 1.0) calculé par l’Analyse Métrique Interne.
- **imacontour** `(string)` : Contour de poids métrique par rapport à la note précédente ('-' : Plus faible, '=' : Égale, '+' : Plus fort)
- **duration** `(float)` : Durée de la note.
- **duration_frac** `(Fraction)` : Durée de la note sous forme de fraction.
- **duration_fullname** `(string)` : Nom complet de la durée de la note (music21).
- **durationcontour** `(string)` : Contour de la durée par rapport à la note précédente ('-' : Plus courte, '=' : Égale, '+' : Plus longue, la première note a None).
- **IOI** `(float)` : Intervalle de temps entre l'apparition de la note et celle de la note suivante (exprimé en unité de noire, la dernière note a None sauf si un silence suit).
- **IOI_frac** `(Fraction)` : Intervalle IOI représenté sous forme de fraction.
- **IOR** `(float)` : Ratio entre l'IOI de la note actuelle et celui de la note précédente (La première note a None).
- **IOR_frac** `(Fraction)` : Ratio IOR exprimé sous forme de fraction.
- **onsettick** `(int)` : Temps d'apparition de la note en ticks MIDI (le premier tick est 0).
- **beatfraction** `(Fraction)` : Durée de la note relative à la durée d'un battement, exprimée en fraction (Si aucune signature rythmique n'est notée, la valeur est None).
- **beat_str** `(string)` : Numéro du temps dans la mesure où se situe la note.
- **beat_fraction_str** `(Fraction)` : Position relative de la note dans son temps, exprimée en fraction.
- **beat** `(float)` : Position de la note dans la mesure, exprimée en unités de temps.
- **songpos** `(float)` : Position de la note dans la chanson, normalisée entre 0.0 (début) et 1.0 (fin).
- **beatinsong** `(Fraction)` : Position de la note dans la chanson, exprimée en unités de battement (par ex., 27/4 pour une note au 6e temps de la 7e mesure).
- **nextisrest** `(bool)` : Indique si la note est suivie d'un silence (true / false, la dernière note a None).
- **restduration_frac** `(Fraction)` : Durée du silence qui suit la note, exprimée en fraction (La valeur est None si aucun silence ne suit).
- **phrase_ix** `(int)` : Numéro de la phrase musicale à laquelle la note appartient. La première phrase est indexée par 0.
- **phrasepos** `(float)` : Position relative de la note dans sa phrase, entre 0.0 (début de la phrase) et 1.0 (fin de la phrase).
- **phrase_end** `(bool)` : Indique si la note est la dernière de la phrase (true ou false).
- **beatinphrase** `(Fraction)` : Position de la note dans sa phrase, exprimée en unités de battement.
- **beatinphrase_end** `(Fraction)` : Position relative de la note dans la phrase, exprimée en unités de battement, où la dernière note de la phrase commence au premier battement de la dernière mesure.
- **gpr2a_Frankland** `(float)` : Force de la frontière musicale suivant la note, calculée selon la règle GPR 2a de la quantification de la théorie GTTM par Frankland et Cohen.
- **gpr2b_Frankland** `(float)` : Force de la frontière musicale suivant la note, calculée selon la règle GPR 2b de la quantification de la théorie GTTM par Frankland et Cohen.
- **gpr3a_Frankland** `(float)` : Force de la frontière musicale suivant la note, calculée selon la règle GPR 3a de la quantification de la théorie GTTM par Frankland et Cohen.
- **gpr3d_Frankland** `(float)` : Force de la frontière musicale suivant la note, calculée selon la règle GPR 3d de la quantification de la théorie GTTM par Frankland et Cohen.
- **gpr_Frankland_sum** `(float)` : Somme des forces des frontières musicales suivant la note, en combinant les résultats des règles GPR 2a, 2b, 3a et 3d.
- **lbdm_boundarystrength** `(float)` : Force globale de la frontière locale suivant la note, calculée selon le modèle de détection des frontières locales (Cambouropoulos).
- **lbdm_spitch** `(float)` : Force de la frontière de hauteur (pitch) suivant la note, selon le modèle LBDM.
- **lbdm_sioi** `(float)` : Force de la frontière d’intervalle entre les onsets (IOI) suivant la note, selon le modèle LBDM.
- **lbdm_srest** `(float)` : Force de la frontière de silence suivant la note, selon le modèle LBDM.
- **lbdm_rpitch** `(float)` : Degré de changement pour l’intervalle de hauteur suivant la note, selon le modèle LBDM.
- **lbdm_rioi** `(float)` : Degré de changement pour l’intervalle entre onsets (IOI) suivant la note, selon le modèle LBDM.
- **lbdm_rrest** `(float)` : Degré de changement pour le silence suivant la note, selon le modèle LBDM.
- **pitchproximity** `(int)` : Expectative de la note basée sur la proximité de hauteur, calculée selon le facteur 1 de la réduction bi-factorielle de l’IR de Narmour par Schellenberg.
- **pitchreversal** `(float)` : Expectative de la note basée sur l’inversion des hauteurs, calculée selon le facteur 2 de la réduction bi-factorielle de l’IR de Narmour par Schellenberg.

> - **melismastate** `(string)` : Indique si la note fait partie d'un mélisme (plusieurs notes pour une syllabe)
> - **lyrics** `(string)` : Syllabe de la parole associée à la note.
> - **noncontentword** `(bool)` : Indique si le mot est un mot fonctionnel (non signifiant) dans la langue néerlandaise.
> - **wordend** `(bool)` : Indique si la syllabe est la dernière (ou unique) d'un mot. Cette donnée est utilisée uniquement pour la première note d’un mélisme vocal.
> - **wordstress** `(bool)` : Indique si la syllabe est accentuée dans le mot. Elle s'applique uniquement à la première note d’un mélisme vocal.
> - **phoneme** `(string)` : Représentation phonétique de la syllabe associée à la note, utilisée uniquement pour la première note d’un mélisme vocal.
> - **rhymes** `(bool)` : Indique si le mot qui se termine sur cette note rime avec un autre mot dans les paroles de la chanson (Cette donnée est utilisée uniquement pour la première note d’un mélisme vocal).
> - **rhymescontentwords** `(bool)` : Indique si le mot qui se termine sur cette note rime avec un autre mot (en excluant les mots fonctionnels) dans les paroles de la chanson (Elle est utilisée uniquement pour la première note d’un mélisme vocal).

### Modeles 

Pour réaliser se projet, nous allons nous limiter au test d'un nombre limité de modeles. Nous allons travailler sur des modèles vue en cours, et le modèle RandomForestClassifier.

- **KNeighborsClassifier**: Classification par proximité avec les k-plus-proches voisins
- **DecisionTreeClassifier**: Arbres de décision pour diviser les données en classes.
- **SGDClassifier**: Optimisation incrémentale pour les grands ensembles de données.
- **LogisticRegression**: Classificateur linéaire basé sur la régression logistique.
- **SVC**: Utilise des marges maximales pour classifier des données, efficace pour les problèmes non linéaires grâce à des noyaux, mais coûteux en temps et mémoire.
- **GaussianNB**: Naïve Bayes avec distribution gaussienne.
- **RandomForestClassifier**: Forêt d’arbres de décision pour réduire le surapprentissage.

Avant de valider, nous allons réaliser une vérification du temps d'exécution de l'entraînement et du test de ces derniers.

![graphe représentant le temps d'entrainement des modèles](/Visualisation/entrainement.png)

Ce graphique nous permet de constater que les modèles prennent tous en moyenne le même temps pour s'entraîner sur les données. Les modèles KNeighborsClassifier et GaussianNB sont les plus rapides.

![graphe représentant le temps de test des modèles](/Visualisation/test.png)

Ce graphique nous permet de constater que le modèle SVC prend beaucoup plus de temps à exécuter le test. Nous pouvons l'expliquer, car il a une complexité de O(n²).
Afin d'éviter des temps d'exécution trop longs, nous n'allons donc pas le garder pour la suite de nos tests.

## METHODOLOGIE : 

### 1) Exploration des données (exploration.ipynb): 

Après avoir téléchargé les données et extrait les différentes features de chaque séquence, nous avons constaté que chaque séquence est représentée
sous forme de liste Python. Chaque élément de cette liste correspond à une note. L'ensemble de notre jeu de données comprend 60 features, une target appelée `phrase_end` et 
un identifiant unique pour chaque séquence. La target `phrase_end` est une colonne booléenne : elle indique si une note correspond à une fin de phrase (True) ou non (False).

En examinant les données, nous avons aussi constaté que certaines features étaient associées aux lyrics des séquences musicales. Ces features sont: `lyrics`, `noncontentword`, `wordend`, 
`phoneme`, `rhymes`, `rhymescontentwords`, `wordstress` et `melismastate`. Ces features ont plus de 50 % de leur valeur qui sont manquantes, ce qui ne nous permet de les utiliser. 
Nous avons donc choisi de les retirer. Il nous reste donc 52 features.

Certaines autres features présentent des valeurs `None` pour certaines notes. Par exemple, la colonne `diatonicinterval` contient tout le temps une valeur 
`None` pour la première note de chaque séquence, car cette valeur dépend de la note précédente. Ces données restent utilisables, nous les avons donc gardées pour nos analyses.

Nous avons étudié la taille des phrases, que nous définissons comme une suite de notes jusqu’à une fin de phrase. Les phrases contiennent en 
moyenne 11 notes, avec une médiane à 9 notes, un minimum de 1 note et un maximum de 238 notes. 

Nous avons réalisé un graphique pour voir la variabilité des tailles de séquences.

![graphe representant la taille des sequences](/Visualisation/graphe_tailles_seqs.png)

Nous avons pour finir classifié les features qu'ils nous restaient en 3 catégories :

- **numérique** : 'scaledegree', 'imaweigth', 'pitch40', 'midipitch', 'diatonicpitch', 'diatonicinterval', 'chromaticinterval', 'pitchproximity', 'pitchreversal', 'duration', 'onsettick', 'phrasepos', 'phrase_ix', 'songpos', 'IOI', 'IOR', 'beatstrength', 'beat_str', 'beat', 'timesignature', 'gpr2a_Frankland', 'gpr2b_Frankland', 'gpr3a_Frankland', 'gpr3d_Frankland', 'gpr_Frankland_sum', 'lbdm_spitch', 'lbdm_sioi', 'lbdm_srest', 'lbdm_rpitch', 'lbdm_rioi', 'lbdm_rrest', 'lbdm_boundarystrength'
- **catégorielle** : 'scaledegreespecifier', 'tonic', 'mode', 'metriccontour', 'nextisrest', 'duration_fullname', 'imacontour', 'pitch', 'contour3', 'contour5', 'durationcontour'
- **label/target** : 'phrase_end'
- **Fraction** : 'duration_frac', 'beatfraction', 'beatinsong', 'beatinphrase', 'beatinphrase_end', 'IOI_frac', 'beat_fraction_str', 'timesignature', 'restduration_frac', 'IOR_frac'

Les colonnes sous forme de fractions ont été reformattées en valeurs numériques et les colones catégorielles ont été encodées avec la méthode OneHotEncoder.

### 2) créations de sous séquences et préparation des données : 

Le jeu de données contient 18108 séquences, afin de trouver les fins de phrases dedans, nous allons les diviser en sous-séquences plus petites.

Le script de création de sous-séquences parcours toutes les listes du jeu de données, et les divise en listes plus petites de tailles que nous pouvons choisir.
Les séquences ayant toutes des tailles différentes, nous faisons attention à ce que les dernières sous-séquences ne soient pas tronquées.

Exemple pour des sous-séquences de taille 4: 
> [1,2,3,4,5,6,7,8,9,10] -> [1,2,3,4] [5,6,7,8] [7,8,9,10]

Cette méthode nous permet bien de récupérer des listes formant des sous-séquences qui sont toutes de la même taille. En réalisant ce découpage, on perd beaucoup de sous-séquences. 
Nous allons réaliser un décalage d'une valeur qu'on peut aussi définir. 

Exemple pour des sous-séquences de taille 4 avec un décalage de 2 (4/2): 
> [1,2,3,4,5,6,7,8,9,10] -> [1,2,3,4] [3,4,5,6] [5,6,7,8] [7,8,9,10]

Il faut cependant éviter de générer toutes les sous-séquences, car cela génère différents problèmes:
- Augmentation des coûts en termes de calcul et de mémoire. Générer toutes les sous-séquences demande plus d'espace mémoire pour les stocker, de plus cela augmenterait le temps de calculs.
- Forte redondance, les sous-séquences sont très similaires les unes des autres, entraînant des calculs inutiles.
- Overfitting, en faisant de l'apprentissage supervisé avec nos modèles, risque d'apprendre des informations redondantes trop spécifiques.

Il est impossible de donner les listes brutes à nos modèles. Nous allons donc étendre nos listes, afin d'avoir une colonne par note et par features.

Exemple pour 2 sous-séquences: 
> | F1           | F2           | -> | F1a | F1b | F1c | F1d | F2a | F2b | F2c | F2d |
> |--------------|--------------|----|-----|-----|-----|-----|-----|-----|-----|-----|
> | [1, 2, 3, 4] | [A, B, C, D] | -> |  1  |  2  |  3  |  4  |  A  |  B  |  C  |  D  |
> | [3, 4, 5, 6] | [C, D, E, F] | -> |  3  |  4  |  5  |  6  |  C  |  D  |  E  |  F  |

Enfin, le tableau étendu est divisé en deux parties : un jeu d’entraînement et un jeu de test. Le jeu de test représente un tiers de la taille du jeu de données.

Certains modèles n'étant pas capables d'interpréter les valeurs manquantes, nous remplaçons ces dernières par des 0.

### 3) Choix des features:

Pour permettre au modèle de prédire correctement les fins de phrases, nous devons lui fournir les données qui lui sont utiles. Cependant, il n'est pas simple de savoir
quelles sont les features utiles pour prédire une fin de phrase en se basant uniquement sur leur description. Afin de répondre à cette problématique, nous avons testé deux approches.

#### a) Choix arbitraire :
Pour commencer, nous avons choisi de manière arbitraire 4 features numériques pour tester l'implémentation de notre script de division des séquences en sous-séquences.
Nous avons donc sélectionné les features suivantes : 
- "scaledegree", 
- "duration", 
- "midipitch", 
- "beatstrength".

Nous avons choisi comme taille pour nos sous-séquences de quatre avec un décalage de deux. Tous les modèles sont testés avec leurs paramètres par défaut.

![Matrice de confusion pour le RandomForestClassifier](/model_test/results/arbitraire/confusion_matrix_DecisionTreeClassifier.png)

*Matrice de confusion pour le RandomForestClassifier.*

Nous pouvons constater que nos données ne sont pas équilibrées. Ce qui était attendu car, il y a forcément plus de sous-séquences qui 
ne sont pas des fins de phrase que de sous-séquences qui sont des fins de phrases. (Il y a 10 fois plus de sous-séquences qui ne sont pas des fins de phrases (189000) que de sous-séquences qui le sont (19000)). 

À cause de ce déséquilibre, nous ne pouvons pas nous baser sur le f1 score du modèle en "weigthed average", meme si ce score est meilleur que le "macro average". Il ne serait pas pertinent de l'utiliser car, il privilégie la classe majoritaire. 

Nous utilisons donc le f1 score macro average, qui est mieux adapté pour comparer les performances globales
Le f1 score combine les mesures de rappel et de précision, offrant une évaluation des performances du modèle.

Le fichier `results/arbitraire/rsultats.txt` présente les résultats détaillés des rapports de classification.


![graphe montrant les f1 score "macro average" des modèles](/model_test/results/arbitraire/models_f1_scores_visualization.png)
Ce graphe nous permet de constater qu'avec les paramètres par défaut, le modèle RandomForestClassifier est le meilleur, avec un f1 score de 0,79. Le modèle avec le moins bon f1 score (0,65) est le SGDClassifier 

Pour améliorer ces scores, nous pouvons essayer d'équilibrer nos données d'entraînement, pour voir si cela permet de réduire l'impact du déséquilibre.

![graphe montrant les f1 score "macro average" des modèles avec les données d'entrainement équilibrées](/model_test/results/arbitraire/models_f1_scores_balanced.png)

L'équilibrage des données d'entraînement n'a pas permis d'améliorer les résultats, pour la majorité des modèles, il réduit même les scores. On peut l'expliquer par l'augmentation du déséquilibre.
Les modèles ont tendance à classer la majorité des sous-séquences comme n'étant pas des fins de phrases.

![Matrice de confusion pour le RandomForestClassifier avec les données d'entrainement équilibrées](/model_test/results/arbitraire/equilibre/confusion_matrix_2RandomForestClassifier.png)

*Matrice de confusion pour le RandomForestClassifier avec les données d'entrainement équilibrées*

Pour améliorer nos résultats, nous avons exploré une autre méthode plus précise pour la sélection des features : l'analyse de corrélation.


#### b) Matrice de corrélations (correlation/global et correlation/numérique):
L'utilisation de quatre features choisit arbitrairement a montré des limites. Nous avons décidé d'utiliser des matrices de corrélation pour identifier les features les plus significatives.
Ces matrices permettent d’évaluer les relations linéaires entre les features et la cible, en calculant le coefficient de corrélation de Pearson. 
Elles permettent également d’identifier les features fortement corrélées entre elles, ce qui nous permet d'éviter les informations redondantes. Ainsi, cette approche a permis de sélectionner les variables pertinentes, simplifiant le modèle et améliorant ses performances.

Pour réaliser nos matrices de correlations, nous commençons par exploser nos listes, afin d'avoir une ligne par note.
Puis, nous devons reformater les features catégorielles, et les features en fraction en valeurs numériques.

Cela nous permet de générer une matrice de corrélation globale: `correlation/matrice_correlation_global.svg`. 
> Cette matrice est trop volumineuse pour être interpretée et affichée.

Nous allons donc, dans un premier temps, nous concentrer sur les features qui sont numériques ou formatées en fractions.

À partir de la première matrice `correlation/num_corr1.svg`, nous pouvons constater que les features qui possèdent une version et une version en fraction sont fortement corrélées.

Nous avons decider de supprimer les versions sous forme de fraction : 'duration_frac', 'beatfraction', 'IOI_frac', 'beat_fraction_str', 'IOR_frac'.

Nous avons aussi identifié d'autres features avec des corrélations élevées, nous les avons enlevées : "diatonicinterval", "midipitch", "beat_str", car elles sont très fortement corréelles avec d'autres features.

Nous avons continué cette même démarche d'analyse de matrice de corrélations jusqu'à obtenir une matrice de correlation simplifiée.

![matrice de corrélation final](/correlation/corr_final.svg)

Nous sélectionnons les attributs présents dans cette matrice : "duration", "beatinphrase", "restduration_frac", "beatinphrase_end", "IOI", "beatstrength", "gpr2b_Frankland", "gpr_Frankland_sum", "lbdm_srest", "lbdm_boundarystrength", "pitch40", "imaweight"

Dans un premier temps, nous le testons uniquement sur le RandomsForestClassifier, afin de verifier qu'il y ait bien une amélioration des résultats par rapport a la sélection arbitraire.

En constatant une amélioration (augmentation d'environ 20% sur le f1 score), nous décidons de le tester sur les autres modèles.

![graphe des f1 score des modeles](/model_test/results/correlation/models_f1_scores_visualization.png)

Malgré le changement de features, le score pour GaussianNB reste faible comparé aux autres modèles. En regardant sa matrice de confusion, on constate que le modèle a tendance à mal classer les fins de phrases, en les catégorisant comme n'étant pas des fins de phrases. 

Pour finir, nous avons testé sur le modèle RandomForestClassifier, l'ajout des features catégorielles. Nous avons constater que le f1-score diminuait, et que le modele avait tendance à mal classer les fins de phrases. De ce fait, nous n'allons pas ajouter de features catégorielles dans nos features sélectionnées.
Cette baisse de ce score après l'ajout de ces features peut s'expliquer par l'augmentation de la complexité des données due au OneHotEncoder qui crée une colonne par valeur unique présente dans la feature.


### 4) choix de la taille des sous sequences

### 5) choix des hyperparametres des modeles

####

####

####

####

####

####

### 6) Résultats finaux et Discussion


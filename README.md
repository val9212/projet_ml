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

- **KNeighborsClassifier**: Classifie un point en fonction des classes majoritaires de ses k-plus proches voisins, idéal pour des frontières complexes, mais sensible aux dimensions élevées.
- **DecisionTreeClassifier**: Utilise une structure arborescente pour diviser les données selon les caractéristiques les plus informatives, facile à interpréter, mais sujette au surapprentissage.
- **SGDClassifier**: Implémente une descente de gradient stochastique pour des modèles linéaires, efficace pour les grands volumes de données et des flux de données en continu.
- **LogisticRegression**: Modèle linéaire qui prédit des probabilités pour des classes, bien adapté aux problèmes binaires et robustes avec régularisation.
- **SVC**: Utilise des marges maximales pour classifier des données, efficace pour les problèmes non linéaires grâce à des noyaux, mais coûteux en temps et mémoire.
- **GaussianNB**: Classificateur bayésien qui suppose des distributions normales pour chaque caractéristique, rapide et performant avec des hypothèses simplifiées.
- **RandomForestClassifier**: Ensemble d'arbres de décision entraînés sur des sous-échantillons aléatoires, robuste au surapprentissage et performant sur des jeux de données variés.

Avant de valider nous allons realiser une verification du temps d'execution de l'entrainement et du test de ces modeles.

![graphe representant le temps d'entrainement des modeles](/Visualisation/entrainement.png)

Ce graphe nous permet de constater que les modeles prennent tous en moyenne le meme temps pour s'entrainer sur les données. Les modeles KNeighborsClassifier et GaussianNB sont les plus rapide.


![graphe representant le temps de test des modeles](/Visualisation/test.png)

Ce graphe nous permet de constater que le modèle SVC prend beaucoup plus de temps à exécuter le test. Nous pouvons l'expliquer, car il a une complexité de O(n²).
Afin d'éviter des temps d'exécution trop longs, nous n'allons donc pas le garder pour la suite de nos tests.

## METHODOLOGIE : 

### 1) Exploration des données (exploration.ipynb): 

Une fois les données téléchargées, et les features de chaque séquence extraite, nous pouvons constater que chaque
Séquence est représentée sous forme de liste Python. Chaque élément d'une liste correspond à une note. Notre jeu de données 
est composé de 60 features, une target et un identifiant. Notre target est la colonne phrase_end qui donne pour chaque note une 
Valeur booléenne True / False (True si la note est une fin de phrase et false si la note ne l'est pas).

Nous pouvons aussi constater que les dernières features (lyrics, noncontentword, wordend, phoneme, rhymes, rhymescontentwords,
wordstress, melismastate), correspondent aux lyrics et paroles des chansons, plus de la moitié de leurs données sont manquantes.
Nous allons donc les retirer. Il nous reste 52 Feature.

Nous pouvons aussi constater que certaines autres features contiennent pour certaines notes des valeurs None, comme par exemple pour diatonicinterval
Où chaque première est un `None` car cette valeur dépend de la note d'avant. (Nous gardons ces features dans la suite de nos analyses)

Nous pouvons constater que la taille des phrases dans les séquences du jeu de données est très variable, l'analyse de la taille (nombre de notes avant une note fin de phrases) nous permet de voir que les phrases font : 
- En moyenne, 11 notes.
- En moyenne 9 notes. 
- Au maximum 238 notes.
- Au minimum 1 note.

Nous avons aussi analysé la taille des séquences, avec une représentation graphique.

![graphe representant la taille des sequences](/Visualisation/graphe_tailles_seqs.png)

En rentrant plus dans le détail, nous pouvons voir que les séquences font:
- En moyenne 72 notes.
- Au maximum 1231 notes.
- Au minimum 3 notes.

Classification des features catégorielles et numériques qu'il nous reste:

- **numérique** : 'scaledegree', 'imaweigth', 'pitch40', 'midipitch', 'diatonicpitch', 'diatonicinterval', 'chromaticinterval', 'pitchproximity', 'pitchreversal', 'duration', 'onsettick', 'phrasepos', 'phrase_ix', 'songpos', 'IOI', 'IOR', 'beatstrength', 'beat_str', 'beat', 'timesignature', 'gpr2a_Frankland', 'gpr2b_Frankland', 'gpr3a_Frankland', 'gpr3d_Frankland', 'gpr_Frankland_sum', 'lbdm_spitch', 'lbdm_sioi', 'lbdm_srest', 'lbdm_rpitch', 'lbdm_rioi', 'lbdm_rrest', 'lbdm_boundarystrength'
- **catégorielle** : 'scaledegreespecifier', 'tonic', 'mode', 'metriccontour', 'nextisrest', 'duration_fullname', 'imacontour', 'pitch', 'contour3', 'contour5', 'durationcontour'
- **label/target** : 'phrase_end'
- **Fraction** : 'duration_frac', 'beatfraction', 'beatinsong', 'beatinphrase', 'beatinphrase_end', 'IOI_frac', 'beat_fraction_str', 'timesignature', 'restduration_frac', 'IOR_frac'

*Les données sous forme de fractions sont reformatées en valeur numérique*

### 2) créations de sous séquences et préparation des données : 

Le jeu de données contient 18108 séquences, afin de trouver les fins de phrases dedans, nous allons les diviser en sous-séquences.

Le script de création de sous-séquences parcours toutes les listes du jeu de données, et les divise en listes plus petites de tailles choisies.
Les séquences ayant toutes des tailles, nous faisons attention à ce que les dernières sous-séquences ne soient pas tronquées.

Exemple pour des sous-séquences de taille 4: 
> [1,2,3,4,5,6,7,8,9,10] -> [1,2,3,4] [5,6,7,8] [7,8,9,10]

Cette méthode nous permet bien de récupérer des listes formant des sous-séquences en étant toutes de la même taille. Mais, en réalisant
Ce découpage, on perd beaucoup de sous-séquences. Nous allons réaliser un décalage d'une valeur qu'on peut définir. 

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

Nous allons ensuite diviser notre tableau de données étendu en 2, en un jeu d'entraînement et un jeu de test. Le jeu de test faisant 1/3 de la taille du jeu d'entraînement.

### 3) Choix des features:

Pour permettre au modèle de prédire correctement les fins de phrases, nous devons lui fournir les données qui lui sont utiles. Cependant, il n'est pas simple de savoir
quelles sont les features utiles pour prédire une fin de phrase en se basant uniquement sur la description de ses dernières. Afin de répondre à cette problématique, nous avons testé 3 hypotheses.

#### a) Choix arbitraire :
Pour commencer, nous avons choisi de manière arbitraire 4 features numériques pour tester l'implémentation de notre script de division des séquences en sous-séquences.
Nous avons donc sélectionné les features suivantes : "scaledegree", "duration", "midipitch", "beatstrength".
Afin de rendre les données comptables avec tous les modèles, nous remplaçons les valeurs manquantes par des 0.


Pour commencer, nous avons tester avec une taille de 4 sur nos modeles.

![matrice de confusion](/model_test/results/arbitraire/confusion_matrix_DecisionTreeClassifier.png)
*matrice de confussion RandomForestClassifier*

Nous pouvons constater que nos données ne sont pas équilibrées. Ce qui est logique car, il y a forcement plus de sous sequences qui ne sont pas des fin de phrase que de sous sequences qui sont des fins de phrases. (il y a 10 fois plus de sous sequences qui ne sont pas des fins de phrases (189000) que de sous sequences qui le sont (19000)). 
De ce fait, nous ne pouvons pas nous baser sur le f1 score du modele en "weigthed average", ce score est meilleur que le "macro average", car il privilegie la classe majoritaire. 

*Nous utilisons ici le f1 score car il represente une combinaison des mesures de rappel et de precision*

Le fichier `results/arbitraire/résulats` nous donne acces aux details de chaque modele.

![graphe des f1 score des modeles](/model_test/results/arbitraire/models_f1_scores_visualization.png)
Ce graphe nous permet de constater qu'avec les parametres par defaut, le modeles RandomForestClassifier est les meilleurs, avec un score de 0,79.

Pour ameliorer notre score, nous pouvons essayer d'equilibrer nos données d'entrainement, pour voir si cela nous permet une meilleur classification.



#### b) Matrice de corrélations :
L'utilisation de seulement de ces 4 features ne permet pas d'obtenir de bons résultats. Il est donc nécessaire de changer d'approche pour identifier les features ayant un impact significatif sur notre cible.
L'utilisation d'une matrice de corrélation répond à cette problématique. Les matrices de corrélation permettent de mettre en évidence l'impact des attributs sur la cible. Elles permettent également d'identifier les features fortement corrélées entre elles, qui peuvent être supprimées afin d'éviter de biaiser l'apprentissage du modèle.


#### c) RandomForest : 


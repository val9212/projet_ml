# Prédiction de fins de phrases musicales

## OBJECTIFS: 
Une séquence musicale peut être jouée ou écrite, comme on peut énoncer un texte ou l'écrire. Comme un texte, une séquence musicale est
Structurée. Cette structure peut se décrire comme une suite de phrases. De ce fait, il est possible de déterminer toutes les fins 
De phrases à l'aide de nombreux facteurs. Évidemment, nous ne réaliserons pas ce travail à la main en écoutant chaque morceau, nous allons utiliser des corpus annotés. En les utilisant, nous pouvons mettre en place des algorithmes d'apprentissage capables de prédire les fins de phrases.

Ainsi, l'objectif de ce projet est de préparer des données, les explorer, construire et comparer des modèles prédictifs de fin de phrase musicale.

## MATÉRIELS: 

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
- **origin** `(string)`: Indique le chemin du fichier kern dans la collection ESSEN.

Les séquences caractéristiques/features correspondent aux séquences de notes dans une mélodie donnée. Elles renseignent sur les 61 features distinctes: 

- **pitch** : Note en notation musicale (format music21).
- **midipitch** : Valeur MIDI de la hauteur (0 à 108).
- **pitch40** : Hauteur en base 40.
- **contour3** : Contour de la hauteur ('-' : Descend, '=' : Égale, '+' : Monte, None pour la 1ère note).
- **contour5** : Contour détaillé ('--', '-', '=', '+', '++', None pour la 1ère note).
- **diatonicinterval** : Intervalle diatonique par rapport à la note précédente (en degrés d'échelle, None pour la 1ère note).
- **chromaticinterval** : Intervalle chromatique (différence MIDI) par rapport à la note précédente (None pour la 1ère note).
- **tonic** : Tonique de la note (A à G avec altérations).
- **mode** : Mode musical (majeur, mineur, etc.).
- **scaledegree** : Degré dans la tonalité (1 à 7).
- **scaledegreespecifier** : Spécification du degré.
- **diatonicpitch** : Hauteur diatonique de la note.
- **timesignature** : Signature rythmique (fraction, None si absente).
- **beatstrength** : Force métrique de la note (0.0 à 1.0).
- **metriccontour** : Variation de la force métrique ('-', '=', '+').
- **imaweight** : Poids métrique interne (0.0 à 1.0).
- **imacontour** : Variation du poids métrique ('-', '=', '+').
- **duration** : Durée de la note (en temps).
- **duration_frac** : Durée en fraction.
- **duration_fullname** : Nom complet de la durée (ex. 'croche').
- **durationcontour** : Variation de la durée ('-', '=', '+', None pour la 1ère note).
- **IOI** : Temps jusqu’à la note suivante (en noires, None pour la dernière).
- **IOI_frac** : IOI en fraction.
- **IOR** : Ratio des IOI (None pour la 1ère note).
- **IOR_frac** : Ratio des IOI en fraction.
- **onsettick** : Apparition en ticks MIDI (commence à 0).
- **beatfraction** : Durée relative d’un battement (fraction, None si absente).
- **beat_str** : Temps dans une mesure.
- **beat_fraction_str** : Position dans le temps sous forme fractionnaire.
- **beat** : Position dans la mesure (en temps).
- **songpos** : Position dans la chanson (0.0 = début, 1.0 = fin).
- **beatinsong** : Position de la note dans la chanson (en battements, ex. '27/4').
- **nextisrest** : True si un silence suit, False sinon.
- **restduration_frac** : Durée du silence suivant en fraction (None si pas de silence).
- **phrase_ix** : Numéro de la phrase musicale (commence à 0).
- **phrasepos** : Position relative dans la phrase (0.0 à 1.0).
- **phrase_end** : True si dernière note de la phrase.
- **beatinphrase** : Position dans la phrase (en battements).
- **beatinphrase_end** : Position relative de la dernière note dans la phrase (en battements).
- **gpr2a_Frankland** : Force de frontière musicale (GPR 2a).
- **gpr2b_Frankland** : Force de frontière musicale (GPR 2b).
- **gpr3a_Frankland** : Force de frontière musicale (GPR 3a).
- **gpr3d_Frankland** : Force de frontière musicale (GPR 3d).
- **gpr_Frankland_sum** : Somme des forces de frontière musicale.
- **lbdm_boundarystrength** : Force des frontières locales (Modèle LBDM).
- **lbdm_spitch** : Force des frontières de hauteur.
- **lbdm_sioi** : Force des frontières de rythme (onsets).
- **lbdm_srest** : Force des frontières de silence.
- **lbdm_rpitch** : Variation de hauteur.
- **lbdm_rioi** : Variation d’intervalle entre onsets.
- **lbdm_rrest** : Variation de silence.
- **pitchproximity** : Attente basée sur la proximité des hauteurs.
- **pitchreversal** : Attente basée sur l’inversion des hauteurs.

> Features concernant les paroles:
> - **melismastate** : Indique si la note fait partie d’un mélisme (plusieurs notes pour une syllabe).
> - **lyrics** : Syllabe attachée à la note.
> - **noncontentword** : True si mot non significatif (en néerlandais), False sinon.
> - **wordend** : True si la syllabe termine ou constitue un mot.
> - **wordstress** : True si syllabe accentuée.
> - **phoneme** : Phonème attaché à la note.
> - **rhymes** : True si ce mot rime avec un autre dans la chanson.
> - **rhymescontentwords** : True si rime avec un mot excluant les mots non significatifs.


### Modèles 

Pour réaliser ce projet, nous allons nous limiter au test d'un nombre limité de modèles de classification. Nous allons travailler sur des modèles vus en cours, et le modèle RandomForestClassifier.

- **KNeighborsClassifier**: Ce modèle est basé sur les "k plus proches voisins" (k-nearest neighbors). Il classe un échantillon en fonction des classes des *k* voisins les plus proches dans l'ensemble d'entraînement.
- **DecisionTreeClassifier**: Ce modèle est basé sur un arbre de décision. Il construit une structure arborescente où les noeuds représentent des décisions basées sur des caractéristiques de données pour séparer les classes.
- **SGDClassifier**: Ce modèle Utilise la descente de gradient stochastique (SGD) pour classer les données.
- **LogisticRegression**: Ce modèle statistique estime la probabilité qu'une observation appartienne à une classe particulière en utilisant une fonction logistique.
- **SVC (Support Vector Classifier)**: Ce modèle utilise les machines à vecteur de support pour classifier les données en maximisant la marge entre les frontières des classes.
- **GaussianNB**: Ce modèle est un classificateur bayésien naïf basé sur une distribution gaussienne (normale).
- **RandomForestClassifier**: Ce modèle est un ensemble de plusieurs arbres de décision où chaque arbre est entraîné sur un sous-ensemble aléatoire des données et des caractéristiques.

Avant de valider, nous allons réaliser une vérification du temps d'exécution de l'entraînement et du test de ces derniers.

![graphe représentant le temps d'entrainement des modèles](/Visualisation/time_training.png)

Ce graphique nous permet de constater que les modèles prennent tous, en moyenne, le même temps pour s'entraîner sur les données. Les modèles KNeighborsClassifier et GaussianNB sont les plus rapides.

![graphe représentant le temps de test des modèles](/Visualisation/time_test.png)

Ce graphique nous permet de constater que le modèle SVC prend beaucoup plus de temps à exécuter le test. Nous pouvons l'expliquer, car il a une complexité de O(n³).
Afin d'éviter des temps d'exécution trop longs, nous n'allons donc pas le garder pour la suite de nos tests.

## METHODOLOGIE : 

### 1) Exploration des données (exploration.ipynb): 

Après avoir téléchargé les données et extrait les différentes features de chaque séquence, nous avons constaté que chaque séquence est représentée
sous forme de liste Python. Chaque élément de cette liste correspond à une note. L'ensemble de notre jeu de données comprend 60 features, une target appelée `phrase_end` et 
un identifiant unique pour chaque séquence. La target `phrase_end` est une colonne booléenne : elle indique si une note correspond à une fin de phrase (True) ou non (False).

En examinant les données, nous avons aussi constaté que certaines features étaient associées aux lyrics des séquences musicales. Ces features sont: `lyrics`, `noncontentword`, `wordend`, 
`phoneme`, `rhymes`, `rhymescontentwords`, `wordstress` et `melismastate`. Ces features ont plus de 50 % de leur valeur qui sont manquantes, ce qui ne nous permet pas de les utiliser. 
Nous avons donc choisi de les retirer. Il nous reste donc 52 features.

Certaines autres features présentent des valeurs `None` pour certaines notes. Par exemple, la colonne `diatonicinterval` contient tout le temps une valeur 
`None` pour la première note de chaque séquence, car cette valeur dépend de la note précédente. Ces données restent utilisables, nous les avons donc gardées pour nos analyses.

Nous avons étudié la taille des phrases, que nous définissons comme une suite de notes jusqu’à une fin de phrase. Les phrases contiennent en 
moyenne 11 notes, avec une médiane à 9 notes, un minimum de 1 note et un maximum de 238 notes. 

Nous avons réalisé un graphique pour voir la variabilité des tailles de séquences.

![graphe representant la taille des sequences](/Visualisation/graphe_tailles_seqs.png)

Enfin, nous avons classifié les features qu'ils nous restaient en 4 catégories :

- **numérique** : 'scaledegree', 'imaweigth', 'pitch40', 'midipitch', 'diatonicpitch', 'diatonicinterval', 'chromaticinterval', 'pitchproximity', 'pitchreversal', 'duration', 'onsettick', 'phrasepos', 'phrase_ix', 'songpos', 'IOI', 'IOR', 'beatstrength', 'beat_str', 'beat', 'timesignature', 'gpr2a_Frankland', 'gpr2b_Frankland', 'gpr3a_Frankland', 'gpr3d_Frankland', 'gpr_Frankland_sum', 'lbdm_spitch', 'lbdm_sioi', 'lbdm_srest', 'lbdm_rpitch', 'lbdm_rioi', 'lbdm_rrest', 'lbdm_boundarystrength'
- **catégorielle** : 'scaledegreespecifier', 'tonic', 'mode', 'metriccontour', 'nextisrest', 'duration_fullname', 'imacontour', 'pitch', 'contour3', 'contour5', 'durationcontour'
- **label/target** : 'phrase_end'
- **Fraction** : 'duration_frac', 'beatfraction', 'beatinsong', 'beatinphrase', 'beatinphrase_end', 'IOI_frac', 'beat_fraction_str', 'timesignature', 'restduration_frac', 'IOR_frac'

Les colonnes sous forme de fractions ont été reformattées en valeurs numériques et les colones catégorielles ont été encodées avec la méthode OneHotEncoder.

### 2) créations de sous-séquences et préparation des données : 

Le jeu de données contient 18108 séquences. Afin de trouver les fins de phrases, nous allons diviser chaque séquence en sous-séquences plus petites.

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

### 4) Choix des features:

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

- Vrais positifs : 10575 occurrences qui correspondent aux sous-séquences prédites comme des fins de phrases par le modèle et en étant des fins de phrase.
- Vrais négatifs : 180971 occurrences qui correspondent aux sous-séquences prédites comme n'étant pas des fins de phrases par le modèle et qui ne sont pas des fins de phrases. 
- Faux positifs : 8916 occurrences qui correspondent aux sous-séquences prédites comme des fins de phrases par le modèle et mais qui ne le sont pas.
- Faux Négatifs : 8093 occurrences qui correspondent aux sous-séquences prédites comme n'étant pas des fins de phrases mais qui le sont vraiment.

On constate donc que le modèle est bien capable de prédire les séquences de la classe majoritaire (vrais négatif). Le modèle a cependant des difficultés à identifier les fins de phrases (Vrais positifs).

Nous pouvons constater que nos données ne sont pas équilibrées. Ce qui était attendu car, il y a forcément plus de sous-séquences qui 
ne sont pas des fins de phrase que de sous-séquences qui sont des fins de phrases. (Il y a 10 fois plus de sous-séquences qui ne sont pas des fins de phrases (189000) que de sous-séquences qui le sont (19000)).

À cause de ce déséquilibre, nous ne pouvons pas nous baser sur le f1 score du modèle en "weigthed average", meme si ce score est meilleur que le "macro average". Il ne serait pas pertinent de l'utiliser car, il privilégie la classe majoritaire. 

Nous utilisons donc le f1 score macro average, qui est mieux adapté pour comparer les performances globales.
Le f1 score combine les mesures de rappel et de précision, donnant une évaluation des performances du modèle.

Le fichier `results/arbitraire/rsultats.txt` présente les résultats détaillés des rapports de classification.

![graphe montrant les f1 score "macro average" des modèles](/model_test/results/arbitraire/models_f1_scores_visualization.png)
Ce graphe nous permet de constater qu'avec les paramètres par défaut, le modèle RandomForestClassifier est le meilleur, avec un f1 score de 0,79. Le modèle avec le moins bon f1 score (0,65) est le SGDClassifier.

Pour améliorer ces scores, nous pouvons essayer d'équilibrer nos données d'entraînement, pour voir si cela permet de réduire l'impact du déséquilibre.
Pour ce faire, nous prenons autant de sous-séquences qui sont des fins de phrase que de séquences qui ne le sont pas, nos données de test restent inchangées.

![graphe montrant les f1 score "macro average" des modèles avec les données d'entrainement équilibrées](/model_test/results/arbitraire/models_f1_scores_balanced.png)

L'équilibrage des données d'entraînement n'a pas permis d'améliorer les résultats, pour la majorité des modèles, il réduit même les scores. On peut l'expliquer par l'augmentation du déséquilibre.
Les modèles ont tendance à classer la majorité des sous-séquences comme n'étant pas des fins de phrases. 

![Matrice de confusion pour le RandomForestClassifier avec les données d'entrainement équilibrées](/model_test/results/arbitraire/equilibre/confusion_matrix_RandomForestClassifier.png)

*Matrice de confusion pour le RandomForestClassifier avec les données d'entrainement équilibrées*

On constate une réduction du nombre de faux positifs (3021 < 8916),  mais une augmentation du nombre de faux négatifs (29423 > 8093). Il y a aussi une réduction du nombre de vrais négatifs, les vrais positifs eux ont augmenté. Nous constatons la meme chose sur tous les modèles. (résultats complet: `/model_test/results/arbitraire/equilibre`) 

Pour améliorer nos résultats, nous avons exploré une autre méthode plus précise pour la sélection des features : l'analyse de corrélation.

#### b) Matrice de corrélations (correlation/global et correlation/numérique):
L'utilisation de quatre features choisies arbitrairement a montré des limites. Nous avons décidé d'utiliser des matrices de corrélation pour identifier les features les plus significatives.
Ces matrices permettent d’évaluer les relations linéaires entre les features et la cible, en calculant le coefficient de corrélation de Pearson. 
Elles permettent également d’identifier les features fortement corrélées entre elles, ce qui nous permet d'éviter les informations redondantes. Ainsi, cette approche a permis de sélectionner les variables pertinentes, simplifiant le modèle et améliorant ses performances.

Pour réaliser nos matrices de correlations, nous commençons par exploser nos listes, afin d'avoir une ligne par note.
Puis, nous devons reformater les features catégorielles, et les features en fraction en valeurs numériques.

Cela nous permet de générer une matrice de corrélation globale: `correlation/matrice_correlation_global.svg`. 
> Cette matrice est trop volumineuse pour être interpretée et affichée.

Nous allons donc, dans un premier temps, nous concentrer sur les features qui sont numériques ou formatées en fractions.

À partir de la première matrice `correlation/num_corr1.svg`, nous pouvons constater que les features qui possèdent une version numérique et une version en fraction sont fortement corrélées.

Nous avons decider de supprimer les versions sous forme de fraction : 'duration_frac', 'beatfraction', 'IOI_frac', 'beat_fraction_str', 'IOR_frac'.

Nous avons aussi identifié d'autres features avec des corrélations élevées, nous les avons enlevées : "diatonicinterval", "midipitch", "beat_str", car elles sont très fortement corrélées avec d'autres features.

Nous avons continué cette même démarche d'analyse de matrice de corrélations jusqu'à obtenir une matrice de correlation simplifiée. En enlevant aussi les features qui ne sont pas corrélées avec notre target.

![matrice de corrélation final](/correlation/results/corr_final.svg)

Nous sélectionnons les attributs présents dans cette matrice : "duration", "beatinphrase", "restduration_frac", "beatinphrase_end", "beatstrength", "gpr2b_Frankland", "gpr_Frankland_sum", "lbdm_srest", "lbdm_boundarystrength", "pitch40", "imaweight".

Nous avons retiré la feature "phrasepos", car elle identifie directement les fins des phrases en leur donnant la valeur 1.

Dans un premier temps, nous le testons uniquement sur le RandomsForestClassifier, afin de verifier qu'il y ait bien une amélioration des résultats par rapport a la sélection arbitraire.

![Matrice de confusion](/model_test/results/correlation/confusion_matrix_RandomForestClassifier.png)

*Matrice de confusion pour le RandomForestClassifier*

Avec les nouvelles features, on constate une baisse du nombre de faux positifs et faux négatifs, une augmentation des vrais positifs et vrais négatifs.

En constatant une amélioration (augmentation d'environ 20% sur le f1 score), nous décidons de le tester sur les autres modèles.

![graphe des f1 score des modèles](/model_test/results/correlation/models_f1_scores_visualization.png)

Malgré le changement de features, le score pour GaussianNB reste faible comparé aux autres modèles. En regardant sa matrice de confusion, on constate que le modèle a tendance à mal classer les fins de phrases, en les catégorisant comme n'étant pas des fins de phrases. (résultats complet: `/model_test/results/correlation`)

Pour finir, nous avons testé sur le modèle RandomForestClassifier l'ajout des features catégorielles. Nous avons constaté que le f1-score diminuait, et que le modèle avait tendance à mal classer les fins de phrases. De ce fait, nous n'allons pas ajouter de features catégorielles dans nos features sélectionnées.
Cette baisse de ce score après l'ajout de ces features peut s'expliquer par l'augmentation de la complexité des données due au OneHotEncoder qui crée une colonne par valeur unique présente dans la feature.

Nous devons désormais vérifier l'impact de la taille de nos sous-séquences sur les modèles.

### 5) choix de la taille des sous-séquences

Maintenant que nous avons choisi les features que nous souhaitons tester, il nous reste à vérifier l'impact des sous-séquences sur nos modèles.

Pour cette partie, nous allons travailler sur un échantillon de notre jeu de données sélectionné aléatoirement. L'échantillon représente un quart du jeu de données complet.

Pour vérifier nos données, nous allons réaliser une validation croisée, et vérifier la moyenne des scores sortis, en faisant varier la taille des sous-séquences.
Nous testons les tailles : 2, 4, 6, 8, 10, 12
Le décalage entre chaque sous-séquence correspond à la moitié de la taille.

Nous obtenons donc les résultats suivants:

![graphe des scores de chaque modèle en fonction des différentes tailles de sous-séquences](/size/model_size.png)

Ce graphique nous permet de constater que de manière générale, les performances des modèles ont tendance à diminuer lorsque la taille des sous-séquences augmente.
Cependant, tous les modèles ne sont pas autant impactés.

- RandomForestClassifier: Le modèle reste stable malgré le changement de la taille des sous-séquences, avec des scores constamment élevés et une faible variation. Il semble bien adapté à toutes les tailles de sous-séquences.
- DecisionTreeClassifier: Le modèle réagit de manière similaire au RandomForestClassifier. Il a un score légèrement plus faible et une petite baisse pour les grandes tailles de sous-séquences.
- KNeighborsClassifier: Le modèle montre une baisse progressive de performance quand la taille des sous-séquences augmente. Il est performant sur les petites tailles de sous-séquences, moins sur les grades tailles.
- SGDClassifier: Le modèle est très instable, il y a de fortes variations en fonction de la taille des sous-séquences. 
- LogisticRegression: Le modèle est stable pour les petites tailles de sous-séquences, mais ses performances diminuent légèrement avec l'augmentation de la taille des sous-séquences.
- GaussianNB: Le modèle est très dépendant de la taille des sous-séquences. Les scores chutent fortement dès que la taille des sous-séquences augmente.

Les modèles étant plus performants sur des petites tailles de sous-séquences, nous allons sélectionner une taille de sous-séquences de 4.
Nous ne choisissons pas de générer des sous-séquences de taille de 2 avec un décalage de 1, car cela reviendrait à générer toutes les sous-séquences possibles, ce qui générerait beaucoup de redondance dans les données.

### 6) Choix des hyperparamètres des modèles

Il est possible d'améliorer les performances de nos modèles en ajustant leurs hyperparamètres. Ces derniers permettent également de réduire le risque de sur-apprentissage.

Les hyperparamètres sont des paramètres spécifiques au modèle que nous définissons manuellement, car ils ne sont pas appris à partir des données. 
Chaque modèle possède ses propres hyperparamètres, mais il n'est pas toujours possible ou nécessaire de tous les tester.

Pour optimiser les hyperparamètres, une méthode couramment utilisée est GridSearchCV(). Cette fonction teste toutes les combinaisons d'hyperparamètres que nous lui fournissons et effectue une validation croisée pour évaluer la performance de chaque configuration.
Nous réalisons le test avec un quart du jeu de données.

#### RandomForestClassifier

Voici la liste des hyperparamètres du RandomForestClassifier que nous allons tester:

- n_estimators: [100, 200, 500] -> Définit le nombre maximal d'arbres dans la forêt.
- max_depth: [10, 20, None] -> Spécifie la profondeur maximale des arbres (None indique qu'il n'y a pas de limite).
- criterion: ['gini', 'entropy'] -> Critère utilisé pour évaluer la qualité de la séparation des branches dans les arbres.
- class_weight: ['balanced', None] -> Permet de gérer les problèmes liés aux données déséquilibrées en ajustant les poids des classes.

Les meilleurs hyperparamètres que nous avons trouvés sont: 
> class_weight=None, criterion='entropy', max_depth=None, n_estimators=200

#### DecisionTreeClassifier

Voici la liste des hyperparamètres du DecisionTreeClassifier que nous allons tester:

- max_depth: [10, 20, None] -> Spécifie la profondeur maximale des arbres (None indique qu'il n'y a pas de limite).
- criterion: ['gini', 'entropy'] -> Critère utilisé pour évaluer la qualité de la séparation des branches dans les arbres.
- min_samples_split: [2, 5, 10] -> Détermine le nombre minimal d'échantillons requis pour diviser un noeud interne.
- class_weight: ['balanced', None] -> Permet de gérer les problèmes liés aux données déséquilibrées en ajustant les poids des classes.

Les meilleurs hyperparamètres que nous avons trouvés sont: 
> class_weight=None, criterion='gini', max_depth=10, min_samples_split=10

#### KNeighborsClassifier

Voici la liste des hyperparamètres du KNeighborsClassifier que nous allons tester:

- n_neighbors: [3, 5, 10, 15] -> Détermine le nombre de voisins à prendre en compte pour la classification.
- weights: ['uniform', 'distance'] -> Spécifie la méthode de pondération des voisins ('uniform' applique un poids égal à tous les voisins, 'distance' attribue des poids inversement proportionnels à la distance).
- metric: ['minkowski', 'euclidean', 'manhattan'] -> Définit la métrique utilisée pour calculer les distances entre les points.
- p: [1, 2] -> Spécifie la puissance utilisée pour la métrique Minkowski. Par exemple, p=1 correspond à la distance de Manhattan, et p=2 correspond à la distance euclidienne.

Les meilleurs hyperparamètres que nous avons trouvés sont: 
> metric='minkowski', n_neighbors=5, p=1, weights='distance'

#### SGDClassifier

Voici la liste des hyperparamètres du SGDCClassifier que nous allons tester:

- loss: ['squared_error','log_loss'] -> Fonction de perte.
- alpha: [0.0001, 0.001, 0.01] -> Coefficient de régularisation pour éviter le surapprentissage.
- class_weight: [None, 'balanced'] -> Permet de gérer les problèmes liés aux données déséquilibrées en ajustant les poids des classes.
- max_iter: [1000, 2000, 5000] -> Nombre maximal d'itérations pour converger.
- tol: [1e-3, 1e-4] -> Tolérance pour le critère d'arrêt de l'optimisation.

Les meilleurs hyperparamètres que nous avons trouvés sont: 
> alpha=0.0001, class_weight=None, loss='log_loss', max_iter=1000, tol=0.001

#### LogisticRegression

Voici la liste des hyperparamètres du LogisticRegression que nous allons tester:

- penalty': ['l1', 'l2'] -> Type de régularisation (l1 pour Lasso, l2 pour Ridge).  
- 'C': [0.01, 0.1, 1, 10] -> Inverse de la force de régularisation (plus la valeur est élevée, plus on réduit la pénalisation).
- class_weight: [None, 'balanced'] -> Permet de gérer les problèmes liés aux données déséquilibrées en ajustant les poids des classes.
- 'solver': ['liblinear', 'lbfgs'] -> Algorithme d'optimisation (liblinear rapide, lbfgs pour grandes données).

Les meilleurs hyperparamètres que nous avons trouvés sont:
> C=10, class_weight=None, penalty='l2', solver='liblinear'

#### GaussianNB

Voici la liste des hyperparamètres du GaussianNB que nous allons tester:

- **var_smoothing**: [1e-9, 1e-8, 1e-7, 1e-6] -> Ajoute une petite valeur (lissée) à la variance pour éviter les instabilités numériques.
- **priors**: [None] -> Probabilités a priori des classes (probabilités de chaque classe avant de voir les données).
- 
Les meilleurs hyperparamètres que nous avons trouvés sont:
> priors=None, var_smoothing=1e-06

### 7) Résultats finaux et Discussion

À l'aide des résultats présent dans `model_test/final`, nous pouvons constater que:

![graphe des f1 score des modèles avec hyperparamètres](/model_test/results/final/models_f1_scores_visualization.png)

L'utilisation d'hyperparamètres sur les modèles améliore, dans la plupart des modèles, la classification.

Modèles améliorés: 
Pour le modèle KNeighborsClassifier, on a une augmentation du score macro average (0,82 -> 0,85), de manière générale, le modèle est plus performant.
Pour le modèle SGDClassifier, on constate une forte augmentation du f1 score macro average (0,83 -> 0,88). Le modèle est donc bien plus performant avec les hyperparamètres.
Pour le modèle de LogisticRegression, on constate une forte augmentation du f1 score macro average (0,87 -> 0,91). Le modèle est donc bien plus performant avec les hyperparamètres.

Modèles similaires:
Pour le modèle DecisionTreeClassifier, les résultats restent similaires, même avec la configuration des hyperparamètres.
Pour le modèle GausianNB, les résultats restent similaires, même avec la configuration des hyperparamètres. Ce modèle a toujours les scores les plus bas, il ne semble pas bien adapté à nos données.
Pour le modèle RandomForestClassifier, les résultats restent similaires, même avec la configuration des hyperparamètres.

D'après nos résultats, le modèle RandomForestClassifier est le meilleur, suivi du DecisionTreeClassifier. Les modèles basés sur des arbres de précision sont les plus performants pour réaliser une classification entre deux classes, dont une majoritaire et une minoritaire.
![Matrice de confusion finale RandomForestClassifier](/model_test/results/final/confusion_matrix_RandomForestClassifier.png)

*Matrice de confusion finale RandomForestClassifier* 

La répartition de la matrice de confusion est correcte, même si le modèle a un peu plus de mal à gérer les faux positifs.

![Courbe ROC RandomForestClassfier](/model_test/results/final/roc_curve_RandomForestClassifier.png)

La courbe ROC du modèle indique une performance quasiment optimale, cela s'explique par la présence du déséquilibre entre les classes.
Cependant, elle peut aussi indiquer que le modèle surprend nos données, pour vérifier ça, nous pouvons réaliser une validation croisée.

Résultats de la validation croisée avec 5 plis.
`0.98256255, 0.9841129 , 0.98544748, 0.98476022, 0.98310598`

Nous pouvons constater que les scores sont très proches. Cela montre qu'il n'y a pas de surapprentissage du modèle. Le faible écart type montre aussi qu'il y a une faible variance, cela montre que le modèle est bien précis de manière constante.

Nous pouvons donc arriver à la conclusion que les features sélectionnées permettent aux modèles de très classer nos sous-séquences fin de phrases malgré le déséquilibre.

Nous pouvons essayer d'améliorer la performance des modèles, en essayant plus d'hyperparamètres différents. Nous pouvons aussi essayer de jouer sur la taille des décalages, car ici la taille de décalage testé est toujours 2.
Nous pouvons aussi essayer d'autres méthodes de choix des features, en utilisant par exemple le modèle RandomForestClassifier qui peut nous donner les features les plus importantes. Nous pouvons aussi essayer d'utiliser les features catégorielles en utilisant une autre méthode d'encodage. 
Nous pouvons aussi essayer d'autres modèles de machine learning tels qu'un réseau de neurones.

## Programme et scripts

Pour réaliser ce projet, différents scripts Python ont été réalisés, ils ont été rangés dans différents dossiers, en suivant l'arborescence suivante.

    ├───correlation             # Contient les scripts pour réaliser les matrices de correlations
    │   └───results             # Contient les matrices de corrélations
    ├───hyperparametres         # Contient tous les notebooks permettant de rechercher les meilleurs hyperparamètres pour chaque modèle
    ├───model_test              # Contient les scripts pour tester les modèles
    │   └───results             # Contient tous les résultats des tests réalisés
    │      ├───arbitraire       # Contient les résultats des tests des modèles avec le choix des features arbitraires
    │      │   └───equilibre    # Contient les résultats des tests des modèles avec le choix des features arbitraires équilibrés
    │      ├───correlation      # Contient les résultats des tests des modèles avec le choix des features par corrélation
    │      └───final            # Contient les résultats des tests des modèles avec les hyperparamètres
    ├───size                    # Contient le notebook permettant de tester les différentes tailles 
    └───visualisation           # Contient le notebook de visualisation des données

### Corrélation

Pour réaliser nos matrices de corrélation, nous avons créé deux scripts python.

- `global.py`: Ce script permet de réaliser la matrice de correlation global.
- `numerique.py`: Ce script contient une fonction permettant de créer des matrices de corrélation en sélectionnant les colonnes que nous souhaitons garder.

Les résultats de ses scripts sont placés dans le dossier `results/`

### Hyperparametres

Pour réaliser nos tests sur les hyperparamètres, nous avons réalisé un notebook par modèles.

- `hyper_decisiontree.ipynb`: Test des différents hyperparamètres sur le modèle DecissionTreeClassifier.
- `hyper_gaussiannb.ipynb`: Test des différents hyperparamètres sur le modèle GaussinNB.
- `hyper_kneighboors.ipynb`: Test des différents hyperparamètres sur le modèle KNeighborsClassifier.
- `hyper_logisticregression.ipynb`: Test des différents hyperparamètres sur le modèle LogisticRegression.
- `hyper_randomforest.ipynb`: Test des différents hyperparamètres sur le modèle RandomForestClassifier.
- `hyper_sgdc.ipynb`: Test des différents hyperparamètres sur le modèle SGDC.

### model_test

Pour entraîner les modèles de manière automatique, nous avons développé un programme Python. 

Ce programme Python est capable de préparer les données de notre jeu de données, et de tester différents modèles prédéfinis.
Pour fonctionner, nous devons définir: 
- `size` : la taille des sous-séquences
- `step` : le décalage entre chaque sous-séquence
- `selected_column` : les colonnes sélectionnées
- `models` : la liste de modèles à tester 

Le programme affiche les rapports de classifications, les matrices de confusions et il crée et enregistre pour chaque modèle la matrice de confusions et la courbe ROC.
À la fin, il crée une figure de comparaison des f1 scores (macro average) pour analyser les différences de performances globales des modèles.

Ce programme est composé de 3 fichiers pour fonctionner :
- `main.py`: qui permet d'exécuter le programme.
- `data_processing.py`: qui contient la classe DataProcessor.
> La classe DataProcessor est composée de 4 méthodes:
> - load_data -> Charge les données brutes depuis le chemin spécifié en utilisant MTCFeatureLoader et les convertit en un DataFrame.
> - clean_data -> Nettoie le jeu de données en retirant les colonnes inutiles et en remplaçant les valeurs manquantes par des zéros.
> - create_subsequences -> Crée des sous-séquences à partir des données, selon une taille et un décalage données.
> - process_features -> Transforme les features et les labels pour en faire des entrées directement utilisables dans un modèle de Machine Learning.
- `models.py`: qui contient la classe DataProcessor.
> La classe ModelTrainer est composée de 4 méthodes:
> - split_data -> Divise les données en ensemble d'entraînement et de test de manière stratifiée.
> - add_model -> Ajoute un modèle à la liste des modèles à entraîner et évaluer. 
> - train_and_evaluate -> Entraîne et évalue tous les modèles ajoutés au ModelTrainer.
> - evaluate_model -> Évalue un modèle unique et stocke les résultats.

- `model_test_step.ipynb`: Ce notebook permet de voir ce que fait le programme étape par étape (seulement sur un seul modèle).

Les résultats de ses scripts sont placés dans le dossier `results/`

### size 

Pour réaliser nos tests sur les tailles de sous-séquences, nous avons réalisé un notebook.

- `model_test_size.ipynb`: Ce notebook permet de réaliser une validation croisée sur différentes tailles de sous-séquences et sur différents modèles. Il nous donne un graphe d'évolution de la performance des modèles en fonction de la taille des sous-séquences.
- `model_size.png`: La figure des résultats.

### Visualisation

Pour réaliser nos analyses sur le jeu de données, nous avons réalisé un notebook.

- `selection_données.ipynb`: Ce notebook permet de réaliser les analyses de base sur le jeu de données.
- `graphe_tailles_seqs.png`: La figure de la variation de la taille des séquences.
- `time_test.png`: La figure montrant le temps d'application du test des modèles.
- `time_training.png`: La figure montrant le temps d'entrainement des modèles.

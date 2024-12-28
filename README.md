Prédiction de fins de phrases musicales

# OBJECTIFS : 
Une pièce musicale peut être jouée ou écrite comme on peut énoncer un texte ou l’écrire. Comme un texte, la pièce musicale est
structurée. Cette structure peut se décrire comme une suite de phrases. De ce fait, il est possible de déterminer toutes les fins 
de phrases à l'aide de nombreux facteurs. Evidemment nous ne réliserons pas ce travail à la main en écoutant chaque morceau. 
En utilisant des corpus annotés, nous pouvons mettre en place des algorithmes d’apprentissage de ces fins de phrase.

Ainsi l'objectif de ce projet est de préparer des données, les explorer, construire et comparer des modèles prédictifs de fin de phrase musicale.


# MATERIELS : 

Pour répondre aux objectifs de ce projet, nous avons récupéré des données issues du Meertens Tune Collections 1. C’est
un corpus de mélodies des Pays-Bas annotés par un ensemble d'attributs : 
Un objet mélodique contient des champs de métadonnées et plusieurs séquences de valeurs de caractéristiques.


Les champs de métadonnées sont : 

	identifiant (string): Identifiant de la mélodie.
	type ({vocal, instrumental}): Type de la chanson. vocal: a des paroles, instrumental: n'a pas de paroles.
	année (int) : Année de publication de la chanson
	famille tune (string) : Identifiant de la famille de syntonisations
	tunefamily_full (string): Nom complet de la famille tune
	freemètre (bool) : Si la mélodie a un (i.e libre pas de noted) mètre.
	ann_bgcorpus (bool) : Les chansons de MTC-FS-INST-2.0 pour lesquelles cette valeur est True n'ont aucun rapport avec les chansons de MTC-ANN-2.0.1.
	origine (string) : Le chemin du fichier ** kern dans la collection ESSEN, indiquant principalement l'origine géographique de la mélodie.


Les séquences caractéristiques correspondent aux séquences de notes dans une mélodie donnée. Elles renseignent sur les 72 caractéristiques distinctes : 

	tangage (str) : Pitch de la note en représentation de chaîne telle que définie dans musique21.
	midipitch (int) : Numéro de note MIDI représentant la hauteur.
	pitch40 (int) : Passer base40 représentation.
	contour3 (str) : Contour de la hauteur par rapport à la note précédente. La première note est None.
	contour5 (str) : Contour de la hauteur par rapport à la note précédente, avec des sauts ≥ 3 en midipitch. La première note est None.
	intervalle diatonique (int) : Intervalle diatonique par rapport à la note précédente. La première note est None.
	intervalle chromatique (int) : Différence de midipitch par rapport à la note précédente. La première note est None.
	tonique (str) : Classe de hauteur de la tonique pour la note actuelle.
	mode (str) : Mode de la note actuelle (ex. major, minor, etc.).
	scaledegree (int) : Degré d'échelle du terrain (1-7).
	scaledegreespecifier (str) : Spécificateur du degré d'échelle (ex. P, M, m, etc.).
	diatonicpitch (int) : Pitch diatonique de la note (à partir de la tonique dans l'octave 0).
	timesignature (str) : Signature horaire pour la note actuelle (n/d ou None si absente).
	force de battement (float) : Poids métrique au moment de l'apparition de la note. None si absente.
	métriquecontour (str) : Contour du poids métrique par rapport à la note précédente. Première note est None.
	poids ima (float) : Poids métrique calculé par l'analyse métrique interne.
	imacontour (str) : Contour du poids métrique (IMA) par rapport à la note précédente. Première note est None.
	durée (float) : Durée de la note (quarternote = 1.0).
	durée_frac (str) : Durée de la note en fraction.
	durée_fullname (str) : Nom complet de la durée.
	duréecontour (str) : Comparaison de durée avec la note précédente. Première note est None.
	IOI (float) : Intervalle de temps entre les débuts de notes successifs. Dernière note est None.
	IOI_frac (str) : IOI en fraction.
	IOR (float) : IOI relatif à la note précédente. Première et dernière note sont None.
	IOR_frac (str) : IOR en fraction.
	début (int) : Début de la note en tiques MIDI.
	beatfraction (str) : Durée de la note relative au temps.
	beat_str (str) : Temps dans la mesure (1er temps = 1).
	beat_fraction_str (str) : Position relative de la note dans le temps.
	battre (float) : Position en unités de battement.
	songpos (float) : Position relative de la note dans la chanson (0.0-1.0).
	beatinsong (str) : Position de la note en unités de battement dans la chanson.
	nextisrest (bool) : Indique si la note est suivie d'un repos.
	restduration_frac (str) : Durée du repos suivant, si présent.
	phrase_ix (int) : Numéro de série de la phrase contenant la note.
	phrasepos (float) : Position de la note dans la phrase (0.0-1.0).
	phrase_fin (bool) : Si la note est la dernière d'une phrase.
	phrase béatine (str) : Apparition de la note dans la phrase en unités de battement.
	beatinphrase_end (str) : Position dans la phrase en battement à partir de la fin.
	mélismastate (str) : Rôle de la note dans un mélisme (start, in, end).
	paroles (str) : Texte associé à la note (mélodies vocales uniquement).
	non-contenu (bool) : Si les paroles sont un mot non contenu (mélodies vocales uniquement).
	mot-minute (bool) : Si la syllabe est la dernière du mot.
	stress de mots (bool) : Si la syllabe est accentuée.
	phonème (str) : Phonème associé à la syllabe.
	rimes (bool) : Si le mot se termine par une rime.
	rimescontentwords (bool) : Si le mot se termine par une rime avec un mot de contenu.
	gpr2a_Frankland (float) : Force de frontière (GPR 2a).
	gpr2b_Frankland (float) : Force de frontière (GPR 2b).
	gpr3a_Frankland (float) : Force de frontière (GPR 3a).
	gpr3d_Frankland (float) : Force de frontière (GPR 3d).
	gpr_Frankland_sum (float) : Somme des forces de frontière GPR.
	lbdm_bounderstrength (float) : Résistance globale des limites locales.
	lbdm_spitch (float) : Force de la limite de hauteur.
	lbdm_sioi (float) : Force de la limite IOI.
	lbdm_srest (float) : Force de la limite de repos.
	lbdm_rpitch (float) : Changement dans l'intervalle de hauteur.
	lbdm_rioi (float) : Changement dans l'intervalle inter-apparition.
	lbdm_rrest (float) : Changement dans le repos suivant.
	pitchproximité (int) : Attente selon le facteur pitchproximity.
	pitchreversal (float) : Attente selon le facteur pitchreversal.


# METHODOLOGIE : 

Sommairement voici les étapes méthodologiques que nous avons déployés : 

0) Selection des modèles de classifications testés : 
	A partir de nos connaissances nous avons selectionnés 7 classifieurs que nous souhaitons comparés : 
		-KneighborsClassifier : Classifie un point en fonction des classes majoritaires de ses k-plus proches voisins, 
					idéal pour des frontières complexes mais sensible aux dimensions élevées.

		-DecisionTreeClassifier : Utilise une structure arborescente pour diviser les données selon les caractéristiques les plus informatives, 
					facile à interpréter mais sujette au surapprentissage.

		-SGDClassifier : 	Implémente une descente de gradient stochastique pour des modèles linéaires, 
					efficace pour des grands volumes de données et des flux de données en continu.

		-LogisticRegression :	Modèle linéaire qui prédit des probabilités pour des classes, 
					bien adapté aux problèmes binaires et robustes avec régularisation.

		-SVC :			Utilise des marges maximales pour classifier des données, 
					efficace pour les problèmes non linéaires grâce à des noyaux mais coûteux en temps et mémoire.

		-GaussianNB : 		Classificateur bayésien qui suppose des distributions normales pour chaque caractéristique, 
					rapide et performant avec des hypothèses simplifiées.

		-RandomForestClassifier :Ensemble d'arbres de décision entraînés sur des sous-échantillons aléatoires, 
					robuste au surapprentissage et performant sur des jeux de données variés.


1) Analyse des séquences de notes par mélodie:
	Sur les 72 caractéristiques, certaines apportent des informations similaires.
	Certaines ne sont pas corrélés ou dépendantes des fins de phrase. 
	Dans cette étape, nous avons selectionné certaines caractéristiques pour poursuivre notre étude.
	Nous avons choisis manuellement de retirer/selectionner les variables ayant un inetret dans notre etude.

2) Analyse des datas : 
	Une chanson est une succession de caractéristiques pour chaque attribut. Le nombre de donnée est variable d'une chanson à une autre mais peut être de tailles très importante 
	rendant complexe notre analyse. Nous avons décidé de réduire en nombre les informations. Pour ce faire nous avons assembler les données par lot de 4 formant des sous séquences. 
	Afin d'analyser les données du mieux possible, nous avons convertit les données en tableau. Chaque colonne représentant un attribut différent et chaque ligne une sous séquence
	(sous forme de liste). 
	Enfin, nous avons normaliser les données afin de pouvoir tester nos modèles sur nos données. 
	
3) Test sur l'ensemble des models 

	On constate un déséquilibre entre le nombre de Positifs et de Négatifs ce qui biaisent nos résultats. Les chiffres semblent indiqué des résultats plutôt corrects mais
	le surnombre de positifs dans notre jeu d'apprentissage font qu'ils ont surappris. Le modèle va préférer associer l'étiquette positif à l'ensemble du jeu plutot que 
	de chercher les négatifs. 

4) Test sur l'ensemble des models en ayant ajusté le jeu d'entrainement de 50% de positif et de 50% de négatif 

	On constate des métriques avec de moins bons résultats mais nous avons effacé le biais issue du surapprentissage des etiquettes positifs. 






# METHODES ALGORITHMIQUES:

Pour répondre aux objectifs de ce projet, nous avons implémenter un code Python. 
Ce code est écrit en programmation orienté objet. Nous avons fait ce choix pour faciliter la lisibilité de ce code et le rendre plus structuré. 
Detail des classes etc.



# RESULTATS INTERMEDIAIRES : 

1) Caractéristiques selectionnés : 
Caractéristiques qui semblent être liés aux fins de phrase (choix manuel) :

	phrase_ix : le numéro de la phras peut fournir un contexte séquentiel important.
	phrasepos : la position de la note dans une phras est directement lié à la fin de phrase.
	beatinphrase_end : Position dans la phrase en battement à partir de la fin. ( caractéristique spécifiquement lié à la fin de phrase) 
	durée : fin de phrase sont souvent plus longues
	beatfraction : durée de la note relative au temps. 
	nextisrest : si la note suivante est un repos, il y a plus de chance que ce soit une fin de phrase
	tonique : Les notes de fin sont souvent dans la tonique
	intervalle diatonique : Compare la tonique d'une note à la tonique de la note précédente
	force de battement : une force métrique peut suggérer un point conclusif
	métriquecontour : un changement de contour peut indiquer une cadence ou une conclusion. 

Carctéristiques qui ont une importance à partir de la méthode Importance_features:

3) Mettre des resultats de ce qui a ete fait. 

	Après avoir fait un premier test sur l'ensemble de notre jeu de donnée nous avons obtenus comme mielleur résultat: 
	
	Pour le test RandomForest: 
	
	Matrice de confusion : 	187257	46413
				7811	17454

	Precision 0.97
	Recall : 0.80
	f1-score : 0.80

4) Test sur l'ensemble des models en ayant ajusté le jeu d'entrainement de 50% de positif et de 50% de négatif 

	Pour le test RandomForest: 
	
	Matrice de confusion : 	9887	1808
				1764	9931

	Precision 0.85
	Recall : 0.85
	f1-score : 0.85

# RESULTAT FINAL : 


# DISCUSSION : 


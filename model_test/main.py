from data_processing import DataProcessor
from models import ModelTrainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def main(window_size, step, selected_columns, models, refactor, cat_columns, norm_column,):
    """
        Programme capable de traiter les données, d'entraîner les modèles et de retourner les résultats liés à ces modèles.


        :param window_size: int
            La taille pour créer les sous-séquences.

        :param step: int
            Le pas (step) pour décaler la fenêtre.

        :param selected_columns: list of str
            Les colonnes de features choisies pour le traitement des données.

        :param models: list of tuple
            Une liste de tuples contenant les objets de modèles et leurs noms respectifs.

        :param refactor: list of int
            Une liste d'index pour les colonnes à reformater (fraction -> float).

        :param cat_columns: list of int
            Une liste d'index pour les colonnes catégorielles à traiter avec OneHotEncoder.

        :param norm_column: list of int
            Une liste d'index pour les colonnes à normaliser.

            :return:
        Cette fonction ne retourne rien, mais génère des visualisations, entraîne les modèles,
        affiche les rapports de classification, les matrices de confusions (PNG), les courbes ROC et un graphique des scores F1 (PNG).
    """
    dataset_path = "MTC-FS-INST-2.0"
    numerical_columns = [ 'scaledegree', 'imaweigth', 'pitch40', 'midipitch', 'diatonicpitch', 'diatonicinterval', 'chromaticinterval', 'pitchproximity', 'pitchreversal', 'duration', 'onsettick', 'phrasepos', 'phrase_ix', 'songpos', 'IOI', 'IOR', 'beatstrength', 'beat_str', 'beat', 'timesignature', 'gpr2a_Frankland', 'gpr2b_Frankland', 'gpr3a_Frankland', 'gpr3d_Frankland', 'gpr_Frankland_sum', 'lbdm_spitch', 'lbdm_sioi', 'lbdm_srest', 'lbdm_rpitch', 'lbdm_rioi', 'lbdm_rrest', 'lbdm_boundarystrength']
    data_processor = DataProcessor(dataset_path, numerical_columns, selected_columns, refactor, cat_columns, norm_column, window_size, step)
    data_processor.clean_data()
    data_processor.create_subsequences()
    data_processor.process_features()

    # Initialisation du ModelTrainer
    model_trainer = ModelTrainer(
        features=data_processor.features,
        labels=data_processor.labels,
        ids=data_processor.ids,
    )

    # Entraînement et évaluation des modèles
    for model, name in models:
        model_trainer.add_model(model, name)

    # Entraînement et évaluation des modèles
    model_trainer.train_and_evaluate()

    # Collect results and plot comparison
    model_names = []
    f1_scores = []

    for name, result in model_trainer.results.items():
        f1_score = result["classification_report"]["macro avg"]["f1-score"]
        model_names.append(name)
        f1_scores.append(f1_score)

    plt.figure(figsize=(10, 6))
    plt.barh(model_names, f1_scores)

    # Ajout des labels et du titre
    plt.xlabel('F1-Score (macro avg)')
    plt.ylabel('Modèles')
    plt.title('F1-Score (macro avg) pour chaque modèle')
    plt.yticks(fontsize=7)

    # Annotation des scores sur chaque barre
    for index, score in enumerate(f1_scores):
        plt.text(score + 0.01, index, f'{score:.2f}', va='center')

    # Affichage du graphique
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(
        'C:/Users/val92/PycharmProjects/projet_ml/model_test/models_f1_scores_visualization.png', format='png')
    plt.show()


if __name__ == "__main__":
    size = 4
    step = size // 2

    #index des colonnes à reformater après le passage en sous-séquences (index de la colonne dans la liste selected_colums. Exemple si taille = 4, et l'index 3 -> [8, 9, 10, 11] 4*3-1)
    refactor = [4,5,6,7,8,9,10,11,12,13,14,15]
    cat_columns = []
    norm_column = []

    selected_columns = ["duration", "beatinphrase", 'restduration_frac', "beatinphrase_end", "beatstrength", "gpr2b_Frankland", "gpr_Frankland_sum", "lbdm_srest", "lbdm_boundarystrength", "pitch40", 'imaweight']

    models = [
        (KNeighborsClassifier(metric='minkowski', n_neighbors=5, p=1, weights='distance'), "KNeighborsClassifier"),
        (DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=10, min_samples_split= 10), "DecisionTreeClassifier"),
        (SGDClassifier(alpha=0.0001, class_weight=None, loss='log_loss', max_iter=1000, tol=0.001), "SGDClassifier"),
        (LogisticRegression(C=10, class_weight=None, penalty='l2', solver='liblinear'), "LogisticRegression"),
        (GaussianNB(priors=None, var_smoothing=1e-06), "GaussianNB"),
        (RandomForestClassifier(class_weight=None, criterion='gini', max_depth=None, n_estimators=500),"RandomForestClassifier",),
    ]

    main(size, step, selected_columns, models, refactor, cat_columns, norm_column)
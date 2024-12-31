# main.py

import logging
from data_processing import DataProcessor
from models import ModelTrainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def main(window_size, models):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger("Main")

    dataset_path = "MTC-FS-INST-2.0"
    data_processor = DataProcessor(dataset_path, window_size=window_size)
    data_processor.clean_data()
    data_processor.create_subsequences()
    data_processor.preprocess_features()

    # Initialize ModelTrainer
    model_trainer = ModelTrainer(
        features=data_processor.features,
        labels=data_processor.labels,
        ids=data_processor.ids,
    )

    for model, name in models:
        model_trainer.add_model(model, name)

    # Train and evaluate models
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
    models = [
        (KNeighborsClassifier(), "KNeighborsClassifier"),
        (DecisionTreeClassifier(), "DecisionTreeClassifier"),
        (SGDClassifier(), "SGDClassifier"),
        (LogisticRegression(), "LogisticRegression"),
        (GaussianNB(), "GaussianNB"),
        (RandomForestClassifier(),"RandomForestClassifier",),
    ]
    main(size, models)

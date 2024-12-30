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


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger("Main")

    dataset_path = "MTC-FS-INST-2.0"
    window_size = 8
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

    # Add models to train
    models = [
        (KNeighborsClassifier(), "KNeighborsClassifier"),
        (DecisionTreeClassifier(max_depth=20), "DecisionTreeClassifier"),
        (SGDClassifier(max_iter=1000, tol=1e-3), "SGDClassifier"),
        (LogisticRegression(max_iter=1000), "LogisticRegression"),
        (SVC(), "SVC"),
        (GaussianNB(), "GaussianNB"),
        (RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced"),"RandomForestClassifier",),
    ]

    for model, name in models:
        model_trainer.add_model(model, name)

    # Train and evaluate models
    model_trainer.train_and_evaluate()

    # Collect results and plot comparison
    model_names = []
    f1_scores = []
    for name, result in model_trainer.results.items():
        f1_score = result["classification_report"]["weighted avg"]["f1-score"]
        model_names.append(name)
        f1_scores.append(f1_score)

    # Plot the F1-scores of the models
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, f1_scores, color="skyblue")
    plt.xlabel("Model")
    plt.ylabel("Weighted F1-Score")
    plt.title("Model Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

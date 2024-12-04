# main.py
from phrase_end_predictor import PhraseEndPredictor
from random_forest_predictor import RandomForestPredictor
from knn_predictor import KNNPredictor
from decision_tree_predictor import DecisionTreePredictor
from sgd_predictor import SGDPredictor
from logistic_regression_predictor import LogisticRegressionPredictor
from svc_predictor import SVCPredictor
from gaussian_nb_predictor import GaussianNBPredictor
import matplotlib.pyplot as plt
import numpy as np


def main():
    dataset_path = "MTC-FS-INST-2.0"
    window_size = 8  # Adjust as needed

    # Initialize a base predictor to load and preprocess data once
    base_predictor = PhraseEndPredictor(dataset_path, window_size)
    base_predictor.run()  # Run the full pipeline up to preprocessing

    models = [
        ("Random Forest", RandomForestPredictor(dataset_path, window_size)),
        ("K-Nearest Neighbors", KNNPredictor(dataset_path, window_size)),
        ("Decision Tree", DecisionTreePredictor(dataset_path, window_size)),
        ("SGD Classifier", SGDPredictor(dataset_path, window_size)),
        ("Logistic Regression", LogisticRegressionPredictor(dataset_path, window_size)),
        ("Support Vector Classifier", SVCPredictor(dataset_path, window_size)),
        ("Gaussian Naive Bayes", GaussianNBPredictor(dataset_path, window_size)),
    ]

    accuracies = []
    model_names = []

    for name, model in models:
        print(f"\n{'='*40}\nRunning model: {name}\n{'='*40}")
        # Share data from base predictor
        model.features = base_predictor.features
        model.labels = base_predictor.labels
        model.ids = base_predictor.ids
        model.X_train = base_predictor.X_train
        model.X_test = base_predictor.X_test
        model.y_train = base_predictor.y_train
        model.y_test = base_predictor.y_test

        # No need to run data preprocessing again
        model.train_model()
        model.evaluate_model()
        # Calculate accuracy
        accuracy = model.model.score(model.X_test, model.y_test)
        accuracies.append(accuracy)
        model_names.append(name)

    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, accuracies, color="skyblue")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

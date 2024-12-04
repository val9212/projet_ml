# models.py

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import logging


class ModelTrainer:
    def __init__(self, features, labels, ids):
        """
        Initialize the ModelTrainer.

        Parameters:
        - features: Numpy array of features.
        - labels: Numpy array of labels.
        - ids: Numpy array of song IDs to group data.
        """
        self.features = features
        self.labels = labels
        self.ids = ids
        self.models = []
        self.results = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def split_data(self):
        """
        Split data into training and testing sets, grouped by song ID.
        """
        self.logger.info("Splitting data into training and testing sets...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(
            gss.split(self.features, self.labels, groups=self.ids)
        )
        self.X_train, self.X_test = self.features[train_idx], self.features[test_idx]
        self.y_train, self.y_test = self.labels[train_idx], self.labels[test_idx]
        self.logger.info(f"Training set shape: {self.X_train.shape}")
        self.logger.info(f"Testing set shape: {self.X_test.shape}")

    def add_model(self, model, name):
        """
        Add a model to the list of models to train.

        Parameters:
        - model: The machine learning model instance.
        - name: Name of the model (string).
        """
        self.models.append((model, name))

    def train_and_evaluate(self):
        """
        Train and evaluate all models added to the trainer.
        """
        self.split_data()
        for model, name in self.models:
            self.logger.info(f"Training model: {name}")
            model.fit(self.X_train, self.y_train)
            self.logger.info(f"Model training completed for {name}.")
            self.evaluate_model(model, name)

    def evaluate_model(self, model, name):
        """
        Evaluate a single model and store the results.

        Parameters:
        - model: The trained machine learning model.
        - name: Name of the model (string).
        """
        self.logger.info(f"Evaluating model: {name}")
        y_pred = model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        self.results[name] = {"classification_report": report, "confusion_matrix": cm}
        self.logger.info(f"Evaluation completed for {name}.")

        # Print evaluation metrics
        print(f"\nModel: {name}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(cm)

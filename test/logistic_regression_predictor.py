# logistic_regression_predictor.py

from phrase_end_predictor import PhraseEndPredictor
from sklearn.linear_model import LogisticRegression


class LogisticRegressionPredictor(PhraseEndPredictor):
    def train_model(self):
        """
        Train a Logistic Regression classifier.
        """
        print("Training the Logistic Regression model...")
        self.model = LogisticRegression(
            random_state=42, class_weight="balanced", max_iter=1000
        )
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

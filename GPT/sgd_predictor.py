# sgd_predictor.py

from phrase_end_predictor import PhraseEndPredictor
from sklearn.linear_model import SGDClassifier


class SGDPredictor(PhraseEndPredictor):
    def train_model(self):
        """
        Train a Stochastic Gradient Descent classifier.
        """
        print("Training the SGD classifier...")
        self.model = SGDClassifier(
            random_state=42, class_weight="balanced", max_iter=1000, tol=1e-3
        )
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

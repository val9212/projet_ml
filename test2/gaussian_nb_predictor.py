# gaussian_nb_predictor.py

from phrase_end_predictor import PhraseEndPredictor
from sklearn.naive_bayes import GaussianNB


class GaussianNBPredictor(PhraseEndPredictor):
    def train_model(self):
        """
        Train a Gaussian Naive Bayes classifier.
        """
        print("Training the Gaussian Naive Bayes model...")
        self.model = GaussianNB()
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

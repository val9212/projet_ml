# knn_predictor.py

from phrase_end_predictor import PhraseEndPredictor
from sklearn.neighbors import KNeighborsClassifier


class KNNPredictor(PhraseEndPredictor):
    def train_model(self):
        """
        Train a K-Nearest Neighbors classifier.
        """
        print("Training the K-Nearest Neighbors model...")
        self.model = KNeighborsClassifier()
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

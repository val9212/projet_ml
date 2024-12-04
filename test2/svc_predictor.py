# svc_predictor.py

from phrase_end_predictor import PhraseEndPredictor
from sklearn.svm import SVC


class SVCPredictor(PhraseEndPredictor):
    def train_model(self):
        """
        Train a Support Vector Classifier.
        """
        print("Training the Support Vector Classifier...")
        self.model = SVC(random_state=42, class_weight="balanced", probability=True)
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

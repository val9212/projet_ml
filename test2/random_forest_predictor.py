# random_forest_predictor.py

from phrase_end_predictor import PhraseEndPredictor
from sklearn.ensemble import RandomForestClassifier


class RandomForestPredictor(PhraseEndPredictor):
    def train_model(self):
        """
        Train a Random Forest classifier.
        """
        print("Training the Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
        )
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

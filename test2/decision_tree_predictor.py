# decision_tree_predictor.py

from phrase_end_predictor import PhraseEndPredictor
from sklearn.tree import DecisionTreeClassifier


class DecisionTreePredictor(PhraseEndPredictor):
    def train_model(self):
        """
        Train a Decision Tree classifier.
        """
        print("Training the Decision Tree model...")
        self.model = DecisionTreeClassifier(
            max_depth=20, random_state=42, class_weight="balanced"
        )
        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.")

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay


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
        self.confusion_matrices = []

    def split_data(self):
        """
        Split data into balanced training and testing sets.
        """
        # Combine features and labels into a single array for easier resampling

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.labels,
            test_size= 1/3,
            stratify=self.labels
        )


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
            model.fit(self.X_train, self.y_train)
            self.evaluate_model(model, name)

    def evaluate_model(self, model, name):
        """
        Evaluate a single model and store the results.

        Parameters:
        - model: The trained machine learning model.
        - name: Name of the model (string).
        """
        y_pred = model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)
        cm = confusion_matrix(self.y_test, y_pred)
        self.results[name] = {"classification_report": report, "confusion_matrix": cm, "score": model.score(self.X_test, self.y_test)}

        # Print evaluation metrics
        print(f"\nModel: {name}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(cm)
        self.confusion_matrices.append((cm, name))
        ConfusionMatrixDisplay(cm).plot()
        plt.savefig(f"results/confusion_matrix_{name}.png")
        RocCurveDisplay.from_estimator(model, self.X_test, self.y_test)
        plt.savefig(f"results/roc_curve_{name}.png")

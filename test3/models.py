# models.py
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import logging
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.utils import resample

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
        Split data into balanced training and testing sets.
        """
        print("Balancing and splitting data into training and testing sets...")

        # Combine features and labels into a single array for easier resampling

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features,
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels
        )
        data = np.column_stack((self.X_train, self.y_train))


        # Separate the majority (Y==0) and minority (Y==1) classes
        data_majority = data[data[:, -1] == 0]
        data_minority = data[data[:, -1] == 1]

        # Downsample the majority class to match the minority class size
        if len(data_majority) > len(data_minority):
            data_majority_downsampled = resample(
                data_majority,
                replace=False,
                n_samples=len(data_minority),
                random_state=42
            )
            balanced_data = np.vstack((data_majority_downsampled, data_minority))
        else:
            data_minority_downsampled = resample(
                data_minority,
                replace=False,
                n_samples=len(data_majority),
                random_state=42
            )
            balanced_data = np.vstack((data_majority, data_minority_downsampled))
        # Shuffle the balanced dataset
        np.random.shuffle(balanced_data)

        # Split back into features and labels
        self.X_train = balanced_data[:, :-1]
        self.y_train = balanced_data[:, -1].astype(int)

        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")

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
        self.results[name] = {"classification_report": report, "confusion_matrix": cm, "score": model.score(self.X_test, self.y_test)}
        self.logger.info(f"Evaluation completed for {name}.")

        # Print evaluation metrics
        print(f"\nModel: {name}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(cm)
        ConfusionMatrixDisplay(cm).plot()
        plt.savefig(f"confusion_matrix_2{name}.png")
        RocCurveDisplay.from_estimator(model, self.X_test, self.y_test)
        plt.show()
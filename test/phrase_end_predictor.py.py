# phrase_end_predictor.py

import MTCFeatures
from MTCFeatures import MTCFeatureLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import logging


class PhraseEndPredictor:
    def __init__(self, dataset_path, window_size=8):
        """
        Initialize the PhraseEndPredictor.

        Parameters:
        - dataset_path: Path to the MTC-FS-INST dataset.
        - window_size: Length of the subsequences to be created.
        """
        self.dataset_path = dataset_path
        self.window_size = window_size
        self.data = None
        self.subsequences = None
        self.features = None
        self.labels = None
        self.ids = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.load_data()

    def load_data(self):
        """
        Load data using MTCFeatureLoader.
        """
        fl = MTCFeatureLoader(self.dataset_path)
        seqs = fl.sequences()

        # Convert sequences into a DataFrame
        phrase_data = []
        for x in tqdm(seqs, desc="Processing sequences"):
            phrase_data.append({"id": x["id"], **x["features"]})

        self.data = pd.DataFrame(phrase_data)
        print(f"DataFrame created with shape {self.data.shape}.")

    def check_list_columns(self):
        """
        Identify list-type columns in the DataFrame.
        """
        print("Identifying list-type columns...")
        list_columns = []
        for col in self.data.columns:
            if (
                col != "id"
                and self.data[col].apply(lambda x: isinstance(x, list)).all()
            ):
                list_columns.append(col)
        print(f"List columns identified: {list_columns}")
        return list_columns

    def create_subsequences(self):
        """
        Create subsequences of fixed length from the data.
        """
        print("Creating subsequences...")
        list_columns = self.check_list_columns()

        # Initialize lists to hold subsequences and labels
        subsequences = []
        labels = []
        ids = []

        # Iterate over each song in the dataset
        for idx, row in tqdm(
            self.data.iterrows(), total=self.data.shape[0], desc="Creating subsequences"
        ):
            song_id = row["id"]
            sequence_length = len(row["scaledegree"])

            # Calculate the number of subsequences for this song
            num_subseq = sequence_length // self.window_size

            # Create non-overlapping subsequences
            for i in range(num_subseq):
                start_idx = i * self.window_size
                end_idx = start_idx + self.window_size

                subseq = {}
                for col in list_columns:
                    subseq[col] = row[col][start_idx:end_idx]

                # Check if the subsequence has the correct length
                if len(subseq["scaledegree"]) == self.window_size:
                    # The label is whether the last note in the subsequence is a phrase end
                    label = subseq["phrase_end"][-1]
                    subsequences.append(subseq)
                    labels.append(label)
                    ids.append(song_id)
                else:
                    continue  # Skip incomplete subsequences

        # Convert to DataFrame
        self.subsequences = pd.DataFrame(subsequences)
        self.subsequences["id"] = ids
        self.subsequences["label"] = labels
        print(f"Created {len(self.subsequences)} subsequences.")

    def preprocess_features(self):
        """
        Preprocess features for model training.
        """
        print("Preprocessing features...")

        # Select relevant features
        feature_columns = ["scaledegree", "duration", "midipitch", "beatstrength"]
        feature_columns = [
            col for col in feature_columns if col in self.subsequences.columns
        ]
        print(f"Selected feature columns: {feature_columns}")

        # Initialize list to hold processed feature arrays
        feature_arrays = []

        # Process each subsequence
        for idx, row in tqdm(
            self.subsequences.iterrows(),
            total=self.subsequences.shape[0],
            desc="Processing subsequences",
        ):
            feature_vector = []
            for col in feature_columns:
                # Flatten the list of features into a single vector
                feature_vector.extend(row[col])
            feature_arrays.append(feature_vector)

        # Convert to numpy array
        self.features = np.array(feature_arrays)
        self.labels = np.array(self.subsequences["label"])
        self.ids = np.array(self.subsequences["id"])

        # Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

        print(f"Feature array shape: {self.features.shape}")
        print(f"Labels array shape: {self.labels.shape}")

    def split_data(self):
        """
        Split data into training and testing sets, grouped by song id.
        """
        print("Splitting data into training and testing sets...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(
            gss.split(self.features, self.labels, groups=self.ids)
        )

        self.X_train, self.X_test = self.features[train_idx], self.features[test_idx]
        self.y_train, self.y_test = self.labels[train_idx], self.labels[test_idx]
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Testing set shape: {self.X_test.shape}")

    def train_model(self):
        """
        Train the model. This method should be implemented in the subclass.
        """
        raise NotImplementedError(
            "train_model method should be implemented in subclass."
        )

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        """
        print("Evaluating the model...")
        y_pred = self.model.predict(self.X_test)

        print("\nClassification Report:")
        from sklearn.metrics import classification_report, confusion_matrix

        print(classification_report(self.y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

    def run(self):
        """
        Run the full pipeline.
        """
        self.create_subsequences()
        self.preprocess_features()
        self.split_data()
        self.train_model()
        self.evaluate_model()

# phrase_end_predictor.py

import MTCFeatures
from MTCFeatures import MTCFeatureLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
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
        Load data using MTCFeatureLoader and clean it.
        """
        print("Loading data...")
        fl = MTCFeatureLoader(self.dataset_path)
        seqs = fl.sequences()

        # Convert sequences into a DataFrame
        phrase_data = []
        for x in tqdm(seqs, desc="Processing sequences"):
            phrase_data.append({"id": x["id"], **x["features"]})

        data = pd.DataFrame(phrase_data)
        print(f"DataFrame created with shape {data.shape}.")

        # Clean the dataset by removing metadata and irrelevant columns
        self.clean_dataset(data)

    def clean_dataset(self, data):
        """
        Clean the dataset by removing irrelevant columns.

        Parameters:
        - data: The raw DataFrame.
        """
        print("Cleaning dataset...")

        # Remove columns with a high percentage of missing values
        missing = data.isnull().mean()
        cols_with_missing = missing[missing > 0.5].index.tolist()
        print(f"Dropping columns with >50% missing values: {cols_with_missing}")
        data.drop(columns=cols_with_missing, inplace=True)

        # Keep only columns that are needed
        # You can adjust this list based on domain knowledge
        columns_to_keep = [
            "id",
            "scaledegree",
            "duration",
            "midipitch",
            "beatstrength",
            "phrase_end",
            "pitch",
            "timesignature",
            "keysignature",
            "mode",
        ]
        columns_to_keep = [col for col in columns_to_keep if col in data.columns]
        data = data[columns_to_keep]

        self.data = data
        print(f"Dataset cleaned. New shape: {self.data.shape}")

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
        print(self.subsequences.head())
        print(f"Created {len(self.subsequences)} subsequences.")

    def preprocess_features(self):
        """
        Preprocess features for model training.
        """
        print("Preprocessing features...")

        # Identify feature columns (exclude 'id' and 'label')
        feature_columns = [
            col for col in self.subsequences.columns if col not in ["id", "label"]
        ]
        print(f"Feature columns: {feature_columns}")

        # Initialize list to hold processed feature arrays
        feature_arrays = []
        labels = []
        ids = []

        # Process each subsequence
        for idx, row in tqdm(
            self.subsequences.iterrows(),
            total=self.subsequences.shape[0],
            desc="Processing subsequences",
        ):
            feature_vector = []

            for col in feature_columns:
                values = row[col]

                # Check if values are lists
                if isinstance(values, list):
                    # If values are categorical (non-numeric), encode them
                    if all(isinstance(v, str) for v in values):
                        # Encode categorical features
                        encoder = LabelEncoder()
                        try:
                            encoded_values = encoder.fit_transform(values)
                        except:
                            encoded_values = [0] * len(
                                values
                            )  # Handle unexpected values
                        feature_vector.extend(encoded_values)
                    else:
                        # Assume numerical values
                        feature_vector.extend(values)
                else:
                    # Single value, check if it's categorical
                    if isinstance(values, str):
                        # Encode categorical features
                        encoder = LabelEncoder()
                        try:
                            encoded_value = encoder.fit_transform([values])[0]
                        except:
                            encoded_value = 0
                        feature_vector.append(encoded_value)
                    else:
                        feature_vector.append(values)

            feature_arrays.append(feature_vector)
            labels.append(row["label"])
            ids.append(row["id"])

        # Convert to numpy array
        self.features = np.array(feature_arrays)
        self.labels = np.array(labels)
        self.ids = np.array(ids)

        if np.isnan(self.features).any():
            print("Imputing NaN values with zero.")
            self.features = np.nan_to_num(self.features, nan=0.0)

        # Encode labels
        print("Encoding labels...")
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

        print(f"Feature array shape: {self.features.shape}")
        print(f"Labels array shape: {self.labels.shape}")

        # Standardize features
        print("Standardizing features...")
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        # Feature selection (optional)
        self.select_features()

    def select_features(self):
        """
        Select the best features to train on.
        """
        print("Selecting best features...")

        # Use feature selection techniques, e.g., variance threshold, mutual information, etc.
        from sklearn.feature_selection import (
            VarianceThreshold,
            SelectKBest,
            mutual_info_classif,
        )

        # Remove features with low variance
        selector = VarianceThreshold(threshold=0.0)
        self.features = selector.fit_transform(self.features)
        print(f"Features shape after variance thresholding: {self.features.shape}")

        # Select top k features based on mutual information
        k = min(100, self.features.shape[1])  # Adjust 'k' as needed
        selector = SelectKBest(mutual_info_classif, k=k)
        self.features = selector.fit_transform(self.features, self.labels)
        print(f"Features shape after SelectKBest: {self.features.shape}")

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
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            accuracy_score,
        )

        print(classification_report(self.y_test, y_pred))

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

        # Calculate accuracy
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"\nAccuracy: {accuracy:.4f}")

    def run(self):
        """
        Run the full pipeline.
        """
        self.create_subsequences()
        self.preprocess_features()
        self.split_data()
        self.train_model()
        self.evaluate_model()

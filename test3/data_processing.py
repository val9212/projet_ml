# data_processing.py

import MTCFeatures
from MTCFeatures import MTCFeatureLoader
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DataProcessor:
    def __init__(self, dataset_path, window_size=8):
        """
        Initialize the DataProcessor.

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
        self.logger = logging.getLogger(self.__class__.__name__)
        self.load_data()

    def load_data(self):
        """
        Load data using MTCFeatureLoader and create a DataFrame.
        """
        self.logger.info("Loading data...")
        fl = MTCFeatureLoader(self.dataset_path)
        seqs = fl.sequences()

        # Convert sequences into a DataFrame
        self.logger.info("Converting sequences into a DataFrame...")
        phrase_data = []
        for x in tqdm(seqs, desc="Processing sequences"):
            phrase_data.append({"id": x["id"], **x["features"]})

        self.data = pd.DataFrame(phrase_data)
        self.logger.info(f"DataFrame created with shape {self.data.shape}.")

    def clean_data(self):
        """
        Clean the data by handling missing values and removing unnecessary columns.
        """
        self.logger.info("Cleaning data...")

        # Remove columns with more than 50% missing values
        missing_values = self.data.isnull().mean()
        columns_to_keep = missing_values[missing_values < 0.5].index
        self.data = self.data[columns_to_keep]
        self.logger.info(
            f"Columns kept after removing those with >50% missing values: {list(columns_to_keep)}"
        )

        # Remove rows where critical columns have None in their lists
        critical_columns = ["scaledegree", "phrase_end"]
        for col in critical_columns:
            self.data = self.data[self.data[col].apply(lambda x: None not in x)]
        self.logger.info(
            f"Data shape after removing rows with None in critical columns: {self.data.shape}"
        )

        # Fill missing values in numerical columns
        numerical_columns = ["duration", "midipitch", "beatstrength"]
        for col in numerical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(
                    lambda x: [0 if v is None else v for v in x]
                )

        # Additional cleaning steps can be added here if necessary

    def check_list_columns(self):
        """
        Identify list-type columns in the DataFrame.
        """
        self.logger.info("Identifying list-type columns...")
        list_columns = []
        for col in self.data.columns:
            if (
                col != "id"
                and self.data[col].apply(lambda x: isinstance(x, list)).all()
            ):
                list_columns.append(col)
        self.logger.info(f"List columns identified: {list_columns}")
        return list_columns

    def create_subsequences(self):
        """
        Create subsequences of fixed length from the data.
        """
        self.logger.info("Creating subsequences...")
        list_columns = self.check_list_columns()

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
        self.logger.info(f"Created {len(self.subsequences)} subsequences.")

    def preprocess_features(self):
        """
        Preprocess features for model training.
        """
        self.logger.info("Preprocessing features...")

        # Select relevant features
        feature_columns = ["scaledegree", "duration", "midipitch", "beatstrength"]
        feature_columns = [
            col for col in feature_columns if col in self.subsequences.columns
        ]
        self.logger.info(f"Selected feature columns: {feature_columns}")

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
        self.logger.info("Encoding labels...")
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

        # Standardize features
        self.logger.info("Standardizing features...")
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.features)

        self.logger.info(f"Feature array shape: {self.features.shape}")
        self.logger.info(f"Labels array shape: {self.labels.shape}")

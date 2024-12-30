from MTCFeatures import MTCFeatureLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fractions import Fraction

class DataProcessor:
    def __init__(self, dataset_path, numerical_columns, selected_feature, window_size=8, step=4):
        """
        Initialize the DataProcessor.

        Parameters:
        - dataset_path: Path to the MTC-FS-INST dataset.
        - window_size: Length of the subsequences to be created.
        """
        self.dataset_path = dataset_path
        self.window_size = window_size
        self.step = step
        self.selected_feature = selected_feature
        self.numerical_columns = numerical_columns

        # Prise en compte pour des données specifiques
        self.refactor = [4,5,6,7,8,9,10,11,12,13,14,15]
        self.cat_columns = []
        self.num_columns = []

        self.data = None
        self.subsequences = None
        self.features = None
        self.labels = None
        self.ids = None
        self.load_data()

    def load_data(self):
        """
        Chargement des données en utilisant MTCFeatureLoader et conversion en DataFrame.
        """
        fl = MTCFeatureLoader(self.dataset_path)
        seqs = fl.sequences()


        phrase_data = []
        for x in tqdm(seqs, desc="Processing sequences"):
            phrase_data.append({"id": x["id"], **x["features"]})

        self.data = pd.DataFrame(phrase_data)

    def clean_data(self):
        """
        Clean the data by handling missing values and removing unnecessary columns.
        """
        # Retirer les features lyrics
        x = self.data.keys()
        no_lyrics = x[:54]
        self.data = self.data[no_lyrics]

        # Remplacer les valeurs manquantes par 0
        for col in self.numerical_columns:
            if col in self.data.columns:
                self.data[col] = self.data[col].apply(
                    lambda x: [0 if v is None else v for v in x]
                )

    def create_subsequences(self):
        """
        Créer des sous sequences à partir d'une taille et d'un pas défini.
        """
        subsequences = []
        labels = []
        ids = []
        list_columns = []

        for col in self.data.columns:
            if col != 'id' and self.data[col].apply(lambda x: isinstance(x, list)).all():
                list_columns.append(col)

        for idx, row in tqdm(self.data.iterrows(), total=self.data.shape[0], desc="Creating subsequences"):
            song_id = row['id']
            sequence_length = len(row['scaledegree'])

            if sequence_length == 0:
                continue

            start_idx = 0
            while start_idx + self.window_size <= sequence_length:
                end_idx = start_idx + self.window_size

                subseq = {}
                for col in list_columns:
                    subseq[col] = row[col][start_idx:end_idx]
                if len(subseq['scaledegree']) == self.window_size:
                    label = subseq['phrase_end'][-1]
                    subsequences.append(subseq)
                    labels.append(label)
                    ids.append(song_id)

                start_idx += self.step

            if start_idx < sequence_length:
                end_idx = sequence_length

                subseq = {}
                for col in list_columns:
                    subseq[col] = row[col][start_idx:end_idx]
                if len(subseq['scaledegree']) == self.window_size:
                    label = subseq['phrase_end'][-1]
                    subsequences.append(subseq)
                    labels.append(label)
                    ids.append(song_id)

        self.subsequences = pd.DataFrame(subsequences)
        self.subsequences["id"] = ids
        self.subsequences["label"] = labels

    def process_features(self):

        feature_arrays = []
        for idx, row in tqdm(self.subsequences.iterrows(), total=self.subsequences.shape[0], desc="Processing subsequences"):
            feature_vector = []
            for col in self.selected_feature:
                feature_vector.extend(row[col])
            feature_arrays.append(feature_vector)
            # on étend les sequences.

        self.features = np.array(feature_arrays)
        self.labels = np.array(self.subsequences['label'])
        self.ids = np.array(self.subsequences['id'])

        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)

        for x in self.refactor:
            # Transformation de chaque élément dans la colonne
            self.features[:, x] = [
                float(Fraction(value)) if isinstance(value, str) and '/' in value else
                float(value) if value is not None else 0.0  # Remplacement des None par 0.0
                for value in self.features[:, x]
            ]

        # Répartition des transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(sparse_output=False), self.cat_columns),  # Encodage catégoriel
                ('num', StandardScaler(), self.num_columns)  # Normalisation des valeurs continues
            ],
            remainder='passthrough'
        )

        self.features = preprocessor.fit_transform(self.features)

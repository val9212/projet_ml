# phrase_end_classifier.py

import MTCFeatures
from MTCFeatures import MTCFeatureLoader
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting the phrase end classification script.")

# Load the data using MTCFeatureLoader
logger.info("Loading data...")
fl = MTCFeatureLoader('MTC-FS-INST-2.0')
seqs = fl.sequences()

# Convert sequences into a DataFrame
logger.info("Converting sequences into a DataFrame...")
phrase_data = []
for x in seqs:
    phrase_data.append({
        'id': x['id'],
        **x['features']
    })

data = pd.DataFrame(phrase_data)
logger.info(f"DataFrame created with shape {data.shape}.")

# Display the percentage of missing values in each column
logger.info("Calculating percentage of missing values in each column:")
missing_values = data.isnull().mean() * 100
logger.info(f"\n{missing_values}")

# Select the first 54 columns (excluding lyrics-related features)
columns_to_keep = data.columns[:54]
data = data[columns_to_keep]
logger.info(f"Selected {len(columns_to_keep)} columns to keep.")

# Function to check if a column is a list-type column
def is_list_column(column):
    return column.apply(lambda x: isinstance(x, list)).all()

# Identify all list columns (excluding 'id')
list_columns = [col for col in data.columns if col != 'id' and is_list_column(data[col])]
logger.info(f"Identified {len(list_columns)} list columns to explode: {list_columns}")

# Explode all list columns simultaneously
logger.info("Exploding list columns...")
data = data.explode(list_columns, ignore_index=True)
logger.info(f"Data exploded. New shape: {data.shape}")

# Handle missing values in 'phrase_end' by dropping rows where it's missing
if 'phrase_end' in data.columns:
    data = data.dropna(subset=['phrase_end'])
    logger.info(f"After dropping missing 'phrase_end', data shape: {data.shape}")
else:
    logger.error("'phrase_end' column is missing in the data.")
    exit(1)

# Convert 'phrase_end' to boolean type
data['phrase_end'] = data['phrase_end'].astype(bool)

# List of categorical columns
categorical_columns = ['scaledegreespecifier', 'tonic', 'mode', 'pitch', 'timesignature',
                       'duration_fullname', 'beat_str', 'beat_fraction_str']
# Filter out columns that are not in the DataFrame
categorical_columns = [col for col in categorical_columns if col in data.columns]
logger.info(f"Categorical columns to encode: {categorical_columns}")

# Replace None with 'Unknown' in categorical columns
data[categorical_columns] = data[categorical_columns].fillna('Unknown')

# Identify numerical columns
numerical_columns = data.columns.difference(categorical_columns + ['id', 'phrase_end'])
logger.info(f"Numerical columns: {numerical_columns.tolist()}")

# Convert numerical columns to numeric and fill NaNs
data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric, errors='coerce')
data[numerical_columns] = data[numerical_columns].fillna(0)

# Handle categorical variables using one-hot encoding
logger.info("Performing one-hot encoding on categorical variables...")
data_encoded = pd.get_dummies(data, columns=categorical_columns)
logger.info(f"Data after encoding has shape: {data_encoded.shape}")

# Separate features and target variable
X = data_encoded.drop(['id', 'phrase_end'], axis=1)
y = data_encoded['phrase_end']

# Ensure that each song's data is either in training or testing set
logger.info("Preparing to split data while keeping songs separate...")

# Get unique song IDs
song_ids = data['id'].unique()
logger.info(f"Total unique song IDs: {len(song_ids)}")

# Create a mapping from song ID to all indices (rows) in the DataFrame
song_id_to_indices = data.groupby('id').indices

# Prepare for group split
groups = data['id']

# Split the data using GroupShuffleSplit to ensure songs are not split between training and test sets
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

logger.info(f"Data split into training and testing sets.")
logger.info(f"Training set shape: {X_train.shape}")
logger.info(f"Testing set shape: {X_test.shape}")

# Initialize the model
logger.info("Initializing the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
logger.info("Training the model...")
model.fit(X_train, y_train)
logger.info("Model training completed.")

# Make predictions on the test set
logger.info("Making predictions on the test set...")
y_pred = model.predict(X_test)

# Evaluate the model
logger.info("Evaluating the model...")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Analyze feature importance
logger.info("Analyzing feature importance...")
importances = model.feature_importances_
feature_names = X.columns
feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("\nTop 10 Feature Importances:")
print(feature_importance.head(10))

import pandas as pd
import numpy as np
from MTCFeatures import MTCFeatureLoader
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

fl = MTCFeatureLoader('MTC-FS-INST-2.0')
seqs = fl.sequences()

phrase_data = []
for ii, x in enumerate(seqs):
    phrase_data.append({
        'id': x['id'],
        **x['features']
    })

data = pd.DataFrame(phrase_data)

# Spécifiez la colonne cible et les features
target_column = "phrase_end"  # Nom de la colonne cible (par exemple, prédire les fins de phrase)
X = data.drop(columns=[target_column])  # Toutes les colonnes sauf la cible
y = data[target_column]  # Colonne cible

# 2. Encoder les variables catégorielles
# Identifiez les colonnes catégorielles
categorical_features = X.select_dtypes(include=["object", "category"]).columns

# Encodeur pour les colonnes catégorielles
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# 3. Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraîner un modèle Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Calculer l'importance des features
feature_importances = rf_model.feature_importances_

# Associer les importances aux noms des features
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importances
}).sort_values(by="Importance", ascending=False)

# Afficher les 10 features les plus importantes
print("Top 10 Features Importantes:")
print(importance_df.head(10))

# 6. Visualiser les importances des features
plt.figure(figsize=(10, 6))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.gca().invert_yaxis()  # Inverser l'axe pour avoir les features les plus importantes en haut
plt.title("Importances des Features")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()

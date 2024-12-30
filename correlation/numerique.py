from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fractions import Fraction
from enum import unique
from statistics import correlation
import MTCFeatures
from MTCFeatures import MTCFeatureLoader
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sympy.physics.units import length
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.python.ops.gen_array_ops import OneHot
from tensorflow.python.ops.gen_experimental_dataset_ops import data_service_dataset_v4

fl = MTCFeatureLoader('MTC-FS-INST-2.0')
seqs = fl.sequences()

phrase_data = []
for ii, x in enumerate(seqs):
    phrase_data.append({
        'id': x['id'],
        **x['features']
    })

data = pd.DataFrame(phrase_data)

x = data.keys()
no_lyrics = x[:54]
data = data[no_lyrics]

no_id = ['scaledegree',
 'scaledegreespecifier',
 'tonic',
 'mode',
 'metriccontour',
 'imaweight',
 'pitch40',
 'midipitch',
 'diatonicpitch',
 'diatonicinterval',
 'chromaticinterval',
 'pitchproximity',
 'pitchreversal',
 'nextisrest',
 'restduration_frac',
 'duration',
 'duration_frac',
 'duration_fullname',
 'onsettick',
 'beatfraction',
 'phrasepos',
 'phrase_ix',
 'phrase_end',
 'songpos',
 'beatinsong',
 'beatinphrase',
 'beatinphrase_end',
 'IOI_frac',
 'IOI',
 'IOR',
 'imacontour',
 'pitch',
 'contour3',
 'contour5',
 'beatstrength',
 'beat_str',
 'beat_fraction_str',
 'beat',
 'timesignature',
 'gpr2a_Frankland',
 'gpr2b_Frankland',
 'gpr3a_Frankland',
 'gpr3d_Frankland',
 'gpr_Frankland_sum',
 'lbdm_spitch',
 'lbdm_sioi',
 'lbdm_srest',
 'lbdm_rpitch',
 'lbdm_rioi',
 'lbdm_rrest',
 'lbdm_boundarystrength',
 'durationcontour',
 'IOR_frac']
data = data[no_id]

data = data.explode(column=['scaledegree', 'scaledegreespecifier', 'tonic', 'mode', 'metriccontour',
 'imaweight', 'pitch40', 'midipitch', 'diatonicpitch', 'diatonicinterval', 'chromaticinterval',
 'pitchproximity', 'pitchreversal', 'nextisrest', 'restduration_frac', 'duration', 'duration_frac',
 'duration_fullname', 'onsettick','beatfraction','phrasepos','phrase_ix',
 'phrase_end', 'songpos', 'beatinsong', 'beatinphrase', 'beatinphrase_end', 'IOI_frac', 'IOI', 'IOR', 'imacontour', 'pitch',
 'contour3', 'contour5', 'beatstrength', 'beat_str', 'beat_fraction_str', 'beat', 'timesignature',
 'gpr2a_Frankland', 'gpr2b_Frankland', 'gpr3a_Frankland', 'gpr3d_Frankland', 'gpr_Frankland_sum',
 'lbdm_spitch', 'lbdm_sioi', 'lbdm_srest', 'lbdm_rpitch', 'lbdm_rioi', 'lbdm_rrest', 'lbdm_boundarystrength', 'durationcontour', 'IOR_frac'])

refactor = ['duration_frac', 'beatfraction', 'beatinsong', 'beatinphrase', 'beatinphrase_end', 'IOI_frac',
            'beat_fraction_str', 'timesignature', 'restduration_frac', 'IOR_frac']

for x in refactor:
    # Transformation de chaque élément dans la colonne
    data.loc[:, x] = [
        float(Fraction(value)) if isinstance(value, str) and '/' in value else
        float(value) if value is not None else 0.0  # Remplacement des None par 0.0
        for value in data.loc[:, x]
    ]

cat_columns = ['scaledegreespecifier', 'tonic', 'mode', 'metriccontour', 'nextisrest', 'duration_fullname',
               'phrase_end', 'imacontour', 'pitch', 'contour3', 'contour5', 'durationcontour']

# Matrice 1

data_num = data[[col for col in data.columns if col not in cat_columns]] #On retire les colonnes catégorielles
data_num['phrase_end'] = data['phrase_end']

label = ['phrase_end']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), label),
    ],
    remainder='passthrough'
)

final_features = preprocessor.fit_transform(data_num)

num_data = pd.DataFrame(final_features, columns=preprocessor.get_feature_names_out())
num_corr = num_data.corr()

num_columns = num_corr.shape[1]
fig_width = num_columns * 0.5
fig_height = num_columns * 0.5

plt.figure(figsize=(fig_width, fig_height))
sns.heatmap(num_corr, cmap="coolwarm", annot=False)

plt.title("Matrice de corrélation des caractéristiques", fontsize=16)
plt.savefig("num_corr1.svg", format="svg", bbox_inches="tight")


# Matrice 2

no_frac = ['duration_frac', 'beatfraction', 'IOI_frac',
            'beat_fraction_str', 'IOR_frac']
data_num2 = data_num[[col for col in data_num.columns if col not in no_frac]]

label = ['phrase_end']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), label),# Encodage catégoriel
    ],
    remainder='passthrough'
)

final_features = preprocessor.fit_transform(data_num2)

num_data = pd.DataFrame(final_features, columns=preprocessor.get_feature_names_out())
num_corr = num_data.corr()

num_columns = num_corr.shape[1]
fig_width = num_columns * 0.5
fig_height = num_columns * 0.5

plt.figure(figsize=(fig_width, fig_height))
sns.heatmap(num_corr, cmap="coolwarm", annot=False)

plt.title("Matrice de corrélation des caractéristiques", fontsize=16)
plt.savefig("num_corr2.svg", format="svg", bbox_inches="tight")

# Matrice 3

remove = ["diatonicinterval", "midipitch", "beat_str"]
data_num3 = data_num2[[col for col in data_num2.columns if col not in remove]]

label = ['phrase_end']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), label),
    ],
    remainder='passthrough'
)

final_features = preprocessor.fit_transform(data_num3)

num_data = pd.DataFrame(final_features, columns=preprocessor.get_feature_names_out())
num_corr = num_data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(num_corr, cmap="coolwarm", annot=False)

plt.title("Matrice de corrélation des caractéristiques", fontsize=16)
plt.savefig("num_corr3.svg", format="svg", bbox_inches="tight")

# Matrice 4

remove = ["beat", "diatonicpitch", "IOR", "lbdm_sioi"]
data_num4 = data_num3[[col for col in data_num3.columns if col not in remove]]

label = ['phrase_end']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), label),
    ],
    remainder='passthrough'
)

final_features = preprocessor.fit_transform(data_num4)

num_data = pd.DataFrame(final_features, columns=preprocessor.get_feature_names_out())
num_corr = num_data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(num_corr, cmap="coolwarm", annot=False)

plt.title("Matrice de corrélation des caractéristiques", fontsize=16)
plt.savefig("num_corr4.svg", format="svg", bbox_inches="tight")

# Matrice 5

remove = ["lbdm_rrest", "lbdm_rioi", "lbdm_rpitch", "lbdm_spitch", "gpr2a_Frankland", "gpr3a_Frankland", "gpr3d_Frankland"]
data_num5 = data_num4[[col for col in data_num4.columns if col not in remove]]

label = ['phrase_end']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), label),
    ],
    remainder='passthrough'
)

final_features = preprocessor.fit_transform(data_num5)

num_data = pd.DataFrame(final_features, columns=preprocessor.get_feature_names_out())
num_corr = num_data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(num_corr, cmap="coolwarm", annot=False)

plt.title("Matrice de corrélation des caractéristiques", fontsize=16)
plt.savefig("num_corr5.svg", format="svg", bbox_inches="tight")

#Matrice final

notin = ["phrase_end", "duration", "beatinphrase", 'restduration_frac', "beatinphrase_end", "IOI", "beatstrength", "gpr2b_Frankland", "gpr_Frankland_sum", "lbdm_srest", "lbdm_boundarystrength", "pitch40", 'imaweight']
data_numf = data_num5[[col for col in data_num5.columns if col in notin]]

label = ['phrase_end']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), label),
    ],
    remainder='passthrough'
)

final_features = preprocessor.fit_transform(data_numf)

num_data = pd.DataFrame(final_features, columns=preprocessor.get_feature_names_out())
num_corr = num_data.corr()

plt.figure(figsize=(10,6))
sns.heatmap(num_corr, cmap="coolwarm", annot=False)

plt.title("Matrice de corrélation des caractéristiques", fontsize=16)
plt.savefig("corr_final.svg", format="svg", bbox_inches="tight")
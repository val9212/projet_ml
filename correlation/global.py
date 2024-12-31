from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from fractions import Fraction
from MTCFeatures import MTCFeatureLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Récuperation des données
fl = MTCFeatureLoader('MTC-FS-INST-2.0')
seqs = fl.sequences()

phrase_data = []
for ii, x in enumerate(seqs):
    phrase_data.append({
        'id': x['id'],
        **x['features']
    })

data = pd.DataFrame(phrase_data)

# Suppression des colonnes inutiles
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

# Explosion de toutes les listes de chaque colones
data3 = data.explode(column=['scaledegree', 'scaledegreespecifier', 'tonic', 'mode', 'metriccontour',
 'imaweight', 'pitch40', 'midipitch', 'diatonicpitch', 'diatonicinterval', 'chromaticinterval',
 'pitchproximity', 'pitchreversal', 'nextisrest', 'restduration_frac', 'duration', 'duration_frac',
 'duration_fullname', 'onsettick','beatfraction','phrasepos','phrase_ix',
 'phrase_end', 'songpos', 'beatinsong', 'beatinphrase', 'beatinphrase_end', 'IOI_frac', 'IOI', 'IOR', 'imacontour', 'pitch',
 'contour3', 'contour5', 'beatstrength', 'beat_str', 'beat_fraction_str', 'beat', 'timesignature',
 'gpr2a_Frankland', 'gpr2b_Frankland', 'gpr3a_Frankland', 'gpr3d_Frankland', 'gpr_Frankland_sum',
 'lbdm_spitch', 'lbdm_sioi', 'lbdm_srest', 'lbdm_rpitch', 'lbdm_rioi', 'lbdm_rrest', 'lbdm_boundarystrength', 'durationcontour', 'IOR_frac'])

# Reformatage des données sous forme de fraction
refactor = ['duration_frac', 'beatfraction', 'beatinsong', 'beatinphrase', 'beatinphrase_end', 'IOI_frac',
            'beat_fraction_str', 'timesignature', 'restduration_frac', 'IOR_frac']
for x in refactor:
    # Transformation de chaque élément dans la colonne
    data3.loc[:, x] = [
        float(Fraction(value)) if isinstance(value, str) and '/' in value else
        float(value) if value is not None else 0.0  # Remplacement des None par 0.0
        for value in data3.loc[:, x]
    ]

# Sélection des colones a reformater avec OneHotEncoder
cat_columns = ['scaledegreespecifier', 'tonic', 'mode', 'metriccontour', 'nextisrest', 'duration_fullname',
               'phrase_end', 'imacontour', 'pitch', 'contour3', 'contour5', 'durationcontour']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse_output=False), cat_columns),  # Encodage catégoriel
    ],
    remainder='passthrough'
)


features = preprocessor.fit_transform(data3)
final_features = pd.DataFrame(features, columns=preprocessor.get_feature_names_out())

correlation_matrix = final_features.corr() # Calcul de la matrice de corrélation

# Création et sauvegarde de la heatmap
num_columns = correlation_matrix.shape[1]
fig_width = num_columns * 0.5
fig_height = num_columns * 0.5

plt.figure(figsize=(fig_width, fig_height)) # dimensions du graphique en fonction du nombre de colonnes
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
plt.title("Matrice de corrélation des caractéristiques", fontsize=16)
plt.savefig("matrice_correlation_global.svg", format="svg", bbox_inches="tight") # Sauvegarde en SVG
plt.close()
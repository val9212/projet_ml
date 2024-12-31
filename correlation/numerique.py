import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fractions import Fraction
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from MTCFeatures import MTCFeatureLoader

# Chargement des données
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
data = data.explode(column=['scaledegree', 'scaledegreespecifier', 'tonic', 'mode', 'metriccontour',
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
    data.loc[:, x] = [
        float(Fraction(value)) if isinstance(value, str) and '/' in value else
        float(value) if value is not None else 0.0  # Remplacement des None par 0.0
        for value in data.loc[:, x]
    ]

# Enlever les colonnes catégorielles
cat_columns = ['scaledegreespecifier', 'tonic', 'mode', 'metriccontour',
               'nextisrest', 'duration_fullname', 'imacontour',
               'pitch', 'contour3', 'contour5', 'durationcontour']

data_num = data[[col for col in data.columns if col not in cat_columns]]


def generate_correlation_matrix(data, save_path, label_column="phrase_end"):
    """
    Génère une heatmap de matrice de corrélation et l'enregistre au format SVG.

    :param data: pd.DataFrame
        Le DataFrame contenant le jeu de données d'entrée.

    :param save_path: str
        Le chemin où sera enregistré le fichier SVG de la heatmap.

    :param label_column: str, optionnel
        Le nom de la colonne catégorielle à encoder en one-hot (par défaut "phrase_end").
    """
    # Préparation des données avec one-hot encoding pour les colonnes catégorielles
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(sparse_output=False), [label_column])],
        remainder='passthrough'
    )
    final_features = preprocessor.fit_transform(data)
    final_data = pd.DataFrame(final_features, columns=preprocessor.get_feature_names_out())

    corr_matrix = final_data.corr()  # Calcul de la matrice de corrélation

    # Création et sauvegarde de la heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    plt.title("Matrice de corrélation des caractéristiques", fontsize=16)
    plt.savefig(save_path, format="svg", bbox_inches="tight") # Sauvegarde en SVG
    plt.close()


# Génération: Matrice 1
generate_correlation_matrix(data_num, "num_corr1.svg")
print("matrice 1 générée")

# Génération: Matrice 2
columns_no_frac = ['duration_frac', 'beatfraction', 'IOI_frac', 'beat_fraction_str', 'IOR_frac'] # Colones retirées
data_num2 = data_num[[col for col in data_num.columns if col not in columns_no_frac]]
generate_correlation_matrix(data_num2, "num_corr2.svg")
print("matrice 2 générée")

# Génération: Matrice 3
columns_to_remove_3 = ["diatonicinterval", "midipitch", "beat_str"] # Colones retirées
data_num3 = data_num2[[col for col in data_num2.columns if col not in columns_to_remove_3]]
generate_correlation_matrix(data_num3, "num_corr3.svg")
print("matrice 3 générée")

# Génération: Matrice 4
columns_to_remove_4 = ["beat", "diatonicpitch", "IOR", "lbdm_sioi"] # Colones retirées
data_num4 = data_num3[[col for col in data_num3.columns if col not in columns_to_remove_4]]
generate_correlation_matrix(data_num4, "num_corr4.svg")
print("matrice 4 générée")

# Génération: Matrice 5
final_columns = ["phrase_end", "duration", "beatinphrase", 'restduration_frac', "beatinphrase_end",
                "beatstrength", "gpr2b_Frankland", "gpr_Frankland_sum", "lbdm_srest",
                 "lbdm_boundarystrength", "pitch40", 'imaweight'] # Colones sélectionnées
data_numf = data_num4[[col for col in data_num4.columns if col in final_columns]]
generate_correlation_matrix(data_numf, "corr_final.svg")
print("matrice 5 générée")

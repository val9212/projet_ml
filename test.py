import MTCFeatures
from MTCFeatures import MTCFeatureLoader
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from webcolors import names

import matplotlib.pyplot as plt

def plot_silhouette(results):
    lengths = list(results.keys())
    scores = list(results.values())
    plt.plot(lengths, scores, marker='o')
    plt.xlabel('Longueur des sous-séquences')
    plt.ylabel('Indice de silhouette')
    plt.title('Qualité des clusters en fonction de la longueur')
    plt.show()

def generate_subsequences(data, max_length):
    subsequences = {}
    for length in range(1, max_length + 1):
        subsequences[length] = [
            data.iloc[i:i + length].drop(columns=['fin_phrase']).values.flatten()
            for i in range(1, len(data) - length)
        ]
    return subsequences


def evaluate_clustering(subsequences):
    results = {}
    scaler = StandardScaler()
    for length, seqs in subsequences.items():
        scaled_data = scaler.fit_transform(seqs)
        kmeans = KMeans(n_clusters=2, random_state=42).fit(scaled_data)
        silhouette = silhouette_score(scaled_data, kmeans.labels_)
        results[length] = silhouette
    return results

fl = MTCFeatureLoader('MTC-FS-INST-2.0')
seqs = fl.sequences()
item = seqs.__next__()

phrase_data = []
for ii, x in enumerate(seqs):
    phrase_data.append({
        'id': x['id'],
        **x['features']
    })
df_phrase = pd.DataFrame(phrase_data)
data = df_phrase[:1000]
nouvelle_colonne = [(0, 1) for _ in range(1000)]
data.drop(columns=['id'], inplace=True)

data['fin_phrase'] = nouvelle_colonne
print(data)
evaluate_clustering(generate_subsequences(data, 1))

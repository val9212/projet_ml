from data_processing import DataProcessor
from models import ModelTrainer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt



def main():
    dataset_path = "MTC-FS-INST-2.0"
    window_size = 4
    step = 2
    numerical_columns = [ 'scaledegree', 'imaweigth', 'pitch40', 'midipitch', 'diatonicpitch', 'diatonicinterval', 'chromaticinterval', 'pitchproximity', 'pitchreversal', 'duration', 'onsettick', 'phrasepos', 'phrase_ix', 'songpos', 'IOI', 'IOR', 'beatstrength', 'beat_str', 'beat', 'timesignature', 'gpr2a_Frankland', 'gpr2b_Frankland', 'gpr3a_Frankland', 'gpr3d_Frankland', 'gpr_Frankland_sum', 'lbdm_spitch', 'lbdm_sioi', 'lbdm_srest', 'lbdm_rpitch', 'lbdm_rioi', 'lbdm_rrest', 'lbdm_boundarystrength']
    selected_columns = ["scaledegree", "duration", "midipitch", "beatstrength"]
    data_processor = DataProcessor(dataset_path, numerical_columns, selected_columns, window_size, step)
    data_processor.clean_data()
    data_processor.create_subsequences()
    data_processor.process_features()

    # Initialize ModelTrainer
    model_trainer = ModelTrainer(
        features=data_processor.features,
        labels=data_processor.labels,
        ids=data_processor.ids,
    )

    # Add models to train
    models = [
        (KNeighborsClassifier(), "KNeighborsClassifier"),
        (DecisionTreeClassifier(), "DecisionTreeClassifier"),
        (SGDClassifier(), "SGDClassifier"),
        (LogisticRegression(), "LogisticRegression"),
        (GaussianNB(), "GaussianNB"),
        (RandomForestClassifier(),"RandomForestClassifier",),
        #(SVC(), "SVC")
    ]

    for model, name in models:
        model_trainer.add_model(model, name)

    # Train and evaluate models
    model_trainer.train_and_evaluate()

    # Collect results and plot comparison
    model_names = []
    f1_scores = []
    for name, result in model_trainer.results.items():
        f1_score = result["classification_report"]["macro avg"]["f1-score"]
        model_names.append(name)
        f1_scores.append(f1_score)

    plt.figure(figsize=(10, 6))
    plt.barh(model_names, f1_scores)

    # Ajout des labels et du titre
    plt.xlabel('F1-Score (macro avg)')
    plt.ylabel('Modèles')
    plt.title('F1-Score (macro avg) pour chaque modèle')

    # Annotation des scores sur chaque barre
    for index, score in enumerate(f1_scores):
        plt.text(score + 0.01, index, f'{score:.2f}', va='center')

    # Affichage du graphique
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
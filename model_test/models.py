from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay


class ModelTrainer:
    def __init__(self, features, labels, ids):
        """
        Classe pour entraîner, évaluer et gérer plusieurs modèles de Machine Learning.

        :param features: np.ndarray
            Un tableau numpy contenant les features utilisées pour l'apprentissage.

        :param labels: np.ndarray
            Un tableau numpy de labels correspondant aux features.

        :param ids: np.ndarray
            Un tableau numpy contenant les ID des chansons ou échantillons pour regrouper les données.
        """
        self.features = features
        self.labels = labels
        self.ids = ids
        self.models = []
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.confusion_matrices = []

    def split_data(self):
        """
        Divise les données en ensemble d'entraînement et de test de manière stratifiée.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features,
            self.labels,
            test_size= 1/3,
            stratify=self.labels
        )


    def add_model(self, model, name):
        """
        Ajoute un modèle à la liste des modèles à entraîner et évaluer.

        :param model: Instance de modèle (ex. RandomForest, LogisticRegression)
            Le modèle de Machine Learning à ajouter.

        :param name: str
            Le nom du modèle pour identification dans les résultats.
        """
        self.models.append((model, name))

    def train_and_evaluate(self):
        """
        Entraîne et évalue tous les modèles ajoutés au ModelTrainer.
        """
        self.split_data()
        for model, name in self.models:
            model.fit(self.X_train, self.y_train)
            self.evaluate_model(model, name)

    def evaluate_model(self, model, name):
        """
        Évalue un modèle unique et stocke les résultats.

            - Prédit les labels sur l'ensemble de test.
            - Génère un rapport de classification.
            - Calcule une matrice de confusion.
            - Sauvegarde les courbes ROC et les matrices de confusion.

        :param model: Instance de modèle entraîné.
            Le modèle à évaluer.

        :param name: str
            Le nom du modèle utilisé pour différencier les résultats.
        """
        y_pred = model.predict(self.X_test)
        report = classification_report(self.y_test, y_pred, output_dict=True)

        # Calcul la matrice de confusion
        cm = confusion_matrix(self.y_test, y_pred)
        self.results[name] = {"classification_report": report, "confusion_matrix": cm, "score": model.score(self.X_test, self.y_test)}

        # Print les métriques d'évaluations
        print(f"\nModel: {name}")
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))
        print("Confusion Matrix:")
        print(cm)
        self.confusion_matrices.append((cm, name))

        # Création et enregistrement des graphes
        ConfusionMatrixDisplay(cm).plot()
        plt.savefig(f"results/confusion_matrix_{name}.png")
        RocCurveDisplay.from_estimator(model, self.X_test, self.y_test)
        plt.savefig(f"results/roc_curve_{name}.png")

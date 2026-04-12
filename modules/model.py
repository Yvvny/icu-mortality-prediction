import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None


class MortalityPredictor:
    """
    Class to train and evaluate ICU mortality prediction models.
    """

    def __init__(self, data: pd.DataFrame):
        """
        Initialize predictor with modeling dataframe.

        Args:
            data (pd.DataFrame): Data containing features and mortality target
        """
        self.data = data
        self.X = data.drop("mortality", axis=1)
        self.y = data["mortality"]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def __str__(self):
        """
        Return summary string for predictor object.
        """
        return f"MortalityPredictor with {len(self.data)} samples"

    def __len__(self):
        """
        Return number of rows in dataset.
        """
        return len(self.data)

    def split_data(self, test_size=0.2, random_state=42):
        """
        Split dataset into training and testing sets.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )

    def train_logistic_regression(self):
        """
        Train baseline logistic regression model.
        """
        if self.X_train is None:
            raise Exception("Run split_data() first")

        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_train, self.y_train)
        self.models["logistic_regression"] = model

    def train_logistic_regression_smote(self):
        """
        Train logistic regression model on SMOTE-balanced training data.
        """
        if self.X_train is None:
            raise Exception("Run split_data() first")

        if SMOTE is None:
            raise Exception("Install imbalanced-learn first using: pip install imbalanced-learn")

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train, self.y_train)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_smote, y_train_smote)
        self.models["logistic_regression_smote"] = model

    def train_random_forest(self):
        """
        Train random forest classifier.
        """
        if self.X_train is None:
            raise Exception("Run split_data() first")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(self.X_train, self.y_train)
        self.models["random_forest"] = model

    def evaluate_model(self, model_name):
        """
        Evaluate trained model using accuracy, F1-score, AUROC,
        confusion matrix, and classification report.
        """
        if self.X_test is None:
            raise Exception("Run split_data() first")

        if model_name not in self.models:
            raise Exception(f"Model '{model_name}' not found")

        model = self.models[model_name]
        y_pred = model.predict(self.X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_prob)
        else:
            auc = None

        results = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1_score": f1_score(self.y_test, y_pred),
            "auroc": auc,
            "confusion_matrix": confusion_matrix(self.y_test, y_pred),
            "classification_report": classification_report(self.y_test, y_pred)
        }

        self.results[model_name] = results
        return results
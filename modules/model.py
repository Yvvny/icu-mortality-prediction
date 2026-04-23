import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

try:
    from imblearn.over_sampling import SMOTE, SMOTENC
except ImportError:
    SMOTE = None
    SMOTENC = None

from modules.data_processor import PatientDataProcessor


class MortalityPredictor:
    """
    Class to train and evaluate ICU mortality prediction models.
    """

    def __init__(self, data_source):
        """
        Initialize predictor with either a processor object or modeling dataframe.

        Args:
            data_source (PatientDataProcessor | pd.DataFrame): Processor object
            or modeling dataframe containing features and mortality target
        """
        self.processor = None
        if isinstance(data_source, PatientDataProcessor):
            self.processor = data_source
            data = self.processor.get_model_data()
        else:
            data = data_source

        if data.empty:
            raise ValueError("Model data cannot be empty")
        if "mortality" not in data.columns:
            raise ValueError("Model data must include a 'mortality' column")

        self.data = data.copy()
        self.X = self.data.drop("mortality", axis=1)
        self.y = self.data["mortality"]
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def _require_split_data(self):
        """
        Ensure train-test data has been created before training or evaluation.
        """
        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise RuntimeError("Run split_data() first")

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
        self._require_split_data()

        model = LogisticRegression(max_iter=1000)
        model.fit(self.X_train, self.y_train)
        self.models["logistic_regression"] = model

    def train_logistic_regression_smote(self):
        """
        Train logistic regression model on SMOTE-balanced training data.
        """
        self._require_split_data()

        if SMOTE is None:
            raise ImportError("Install imbalanced-learn first using: pip install imbalanced-learn")

        categorical_features = []
        if "gender_encoded" in self.X_train.columns:
            categorical_features.append(self.X_train.columns.get_loc("gender_encoded"))

        if categorical_features and SMOTENC is not None:
            sampler = SMOTENC(categorical_features=categorical_features, random_state=42)
        else:
            sampler = SMOTE(random_state=42)

        X_train_smote, y_train_smote = sampler.fit_resample(self.X_train, self.y_train)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train_smote, y_train_smote)
        self.models["logistic_regression_smote"] = model

    def train_random_forest(self):
        """
        Train random forest classifier.
        """
        self._require_split_data()

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
        self._require_split_data()

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model = self.models[model_name]
        y_pred = model.predict(self.X_test)

        if hasattr(model, "predict_proba") and self.y_test.nunique() > 1:
            y_prob = model.predict_proba(self.X_test)[:, 1]
            auc = roc_auc_score(self.y_test, y_prob)
        else:
            auc = None

        results = {
            "accuracy": accuracy_score(self.y_test, y_pred),
            "f1_score": f1_score(self.y_test, y_pred),
            "auroc": auc,
            "confusion_matrix": confusion_matrix(self.y_test, y_pred),
            "classification_report": classification_report(self.y_test, y_pred, zero_division=0)
        }

        self.results[model_name] = results
        return results

import matplotlib.pyplot as plt
import pandas as pd


# ==============================
# 1. Mortality Distribution
# ==============================
def plot_mortality_distribution(df, save_path=None):
    """
    Visualize the distribution of mortality outcomes.

    Args:
        df (pd.DataFrame): DataFrame containing a 'mortality' column.
        save_path (str, optional): File path to save the figure.
    """
    if "mortality" not in df.columns:
        raise ValueError("The DataFrame must contain a 'mortality' column.")

    counts = df["mortality"].value_counts().sort_index()

    plt.figure()
    plt.bar(["Survived (0)", "Died (1)"], counts.values)
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.title("Mortality Distribution")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# ==============================
# 2. Model F1 Score Comparison
# ==============================
def plot_model_f1_scores(score_dict, save_path=None):
    """
    Compare F1 scores across different models.

    Args:
        score_dict (dict): Dictionary mapping model names to F1 scores.
        save_path (str, optional): File path to save the figure.
    """
    if not isinstance(score_dict, dict) or len(score_dict) == 0:
        raise ValueError("score_dict must be a non-empty dictionary.")

    models = list(score_dict.keys())
    scores = list(score_dict.values())

    plt.figure()
    plt.bar(models, scores)
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.title("Model Comparison by F1 Score")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# ==============================
# 3. Mortality by Age Group
# ==============================
def plot_mortality_by_age_group(df, save_path=None):
    """
    Visualize mortality rate across different age groups.

    Args:
        df (pd.DataFrame): DataFrame containing 'age_group' and 'mortality'.
        save_path (str, optional): File path to save the figure.
    """
    if "age_group" not in df.columns:
        raise ValueError("The DataFrame must contain an 'age_group' column.")

    if "mortality" not in df.columns:
        raise ValueError("The DataFrame must contain a 'mortality' column.")

    rates = df.groupby("age_group", observed=False)["mortality"].mean()

    plt.figure()
    plt.bar(rates.index.astype(str), rates.values)
    plt.xlabel("Age Group")
    plt.ylabel("Mortality Rate")
    plt.title("Mortality Rate by Age Group")
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# ==============================
# 4. Confusion Matrix Visualization
# ==============================
def plot_confusion_matrix(cm, model_name="Model", save_path=None):
    """
    Display the confusion matrix for a given model.

    Args:
        cm (array-like): Confusion matrix.
        model_name (str): Name of the model.
        save_path (str, optional): File path to save the figure.
    """
    if cm is None:
        raise ValueError("Confusion matrix cannot be None.")

    plt.figure()
    plt.imshow(cm)
    plt.title(f"{model_name} Confusion Matrix")
    plt.colorbar()

    for i in range(len(cm)):
        for j in range(len(cm[0])):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# ==============================
# 5. Feature Importance
# ==============================
def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plot feature importance for tree-based models.

    Args:
        model: Trained model with feature_importances_ attribute.
        feature_names (list): List of feature names.
        save_path (str, optional): File path to save the figure.
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("The model does not support feature_importances_.")

    importances = model.feature_importances_

    plt.figure()
    plt.bar(feature_names, importances)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Feature Importance")
    plt.xticks(rotation=45)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


# ==============================
# 6. Export Model Results
# ==============================
def export_model_results(results_dict, output_path="model_results.csv"):
    """
    Export model evaluation metrics to a CSV file.

    Args:
        results_dict (dict): Dictionary containing model evaluation results.
        output_path (str): Path to save the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the exported results.
    """
    if not isinstance(results_dict, dict) or len(results_dict) == 0:
        raise ValueError("results_dict must be a non-empty dictionary.")

    rows = []
    for name, res in results_dict.items():
        rows.append({
            "model": name,
            "accuracy": res.get("accuracy"),
            "f1_score": res.get("f1_score"),
            "auroc": res.get("auroc"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df
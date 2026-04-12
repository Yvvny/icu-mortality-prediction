import matplotlib.pyplot as plt


def plot_mortality_distribution(df):
    """
    Plot the mortality class distribution.

    Args:
        df (pd.DataFrame): Dataframe containing mortality column
    """
    counts = df["mortality"].value_counts().sort_index()

    plt.figure()
    plt.bar(["Survived (0)", "Died (1)"], counts.values)
    plt.xlabel("Outcome")
    plt.ylabel("Count")
    plt.title("Mortality Distribution")
    plt.show()


def plot_model_f1_scores(score_dict):
    """
    Plot F1-scores for different models.

    Args:
        score_dict (dict): Dictionary with model names as keys and F1 scores as values
    """
    models = list(score_dict.keys())
    scores = list(score_dict.values())

    plt.figure()
    plt.bar(models, scores)
    plt.xlabel("Model")
    plt.ylabel("F1 Score")
    plt.title("Model Comparison by F1 Score")
    plt.show()
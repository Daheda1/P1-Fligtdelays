import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np

def load_test_data(file_path):
    # Indlæser data fra CSV-fil
    data = pd.read_csv(file_path)
    # Antager at "DelayLabel" er kolonnen for labels
    X_test = data.drop('DelayLabel', axis=1)
    y_test = data['DelayLabel']
    return X_test, y_test

def calculate_specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    specificity = []
    for i in range(len(labels)):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificity.append(spec)
    return np.mean(specificity)

def evaluate_model(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, labels=labels, average='macro'),
        "recall": recall_score(y_test, y_pred, labels=labels, average='macro'),
        "f1_score": f1_score(y_test, y_pred, labels=labels, average='macro'),
        "specificity": calculate_specificity(y_test, y_pred, labels)
    }
    return scores

def main(model_files):
    X_test, y_test = load_test_data("Testset.csv")
    labels = sorted(y_test.unique())  # Antager at labels er sorteret numerisk eller alfabetisk
    all_scores = {}

    for file in model_files:
        model = joblib.load(file)
        scores = evaluate_model(model, X_test, y_test, labels)
        all_scores[file] = scores

    with open("model_performance.txt", "w") as f:
        for model_name, scores in all_scores.items():
            f.write(f"Model: {model_name}\n")
            for metric, score in scores.items():
                f.write(f"{metric}: {score}\n")
            f.write("\n")

# Eksempel på brug
model_files = ["KNeighborsClassifier.joblib", "RandomForestClassifier.joblib", "GaussianNB.joblib", "DecisionTreeClassifier.joblib", "SGDClassifier.joblib"]
main(model_files)

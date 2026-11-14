import pandas as pd
from pathlib import Path
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump

ROOT = Path(__file__).resolve().parent.parent
FEATURE_FOLDER = ROOT / "data" / "features"
MODEL_FOLDER = ROOT / "models"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

algorithms = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "NaiveBayes": MultinomialNB(),
    "LinearSVM": LinearSVC()
}

tasks = ["sentiment", "offensive"]

results = []

for task in tasks:

    print(f"\n{'=' * 45}")
    print(f"Training {task.upper()} models")
    print("=" * 45)

    X_train = load_npz(FEATURE_FOLDER / f"{task}_X_train.npz")
    X_dev = load_npz(FEATURE_FOLDER / f"{task}_X_dev.npz")

    y_train = pd.read_csv(
        FEATURE_FOLDER / f"{task}_y_train.csv"
    )["label"]

    y_dev = pd.read_csv(
        FEATURE_FOLDER / f"{task}_y_dev.csv"
    )["label"]

    for name, model in algorithms.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_dev)

        accuracy = accuracy_score(y_dev, predictions)
        f1 = f1_score(
            y_dev,
            predictions,
            average="weighted",
            zero_division=0
        )

        results.append({
            "Task": task,
            "Model": name,
            "Accuracy": accuracy,
            "Weighted_F1": f1
        })

        model_path = MODEL_FOLDER / f"{task}_{name}.joblib"
        dump(model, model_path)

        print("Accuracy:", round(accuracy, 4))
        print("Weighted F1:", round(f1, 4))
        print("Model saved:", model_path)

results_df = pd.DataFrame(results)

results_df.to_csv(
    MODEL_FOLDER / "model_comparison.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print(results_df.to_string(index=False))

print("\nModel training completed!")
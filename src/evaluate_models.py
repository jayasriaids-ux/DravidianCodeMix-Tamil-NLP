import pandas as pd
import joblib
from scipy.sparse import load_npz
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

tasks = ["sentiment", "offensive"]

for task in tasks:

    print("\n" + "=" * 50)
    print(f"{task.upper()} MODEL EVALUATION")
    print("=" * 50)

    # Load test data
    X_test = load_npz(f"data/features/{task}_X_test.npz")
    y_test = pd.read_csv(f"data/features/{task}_y_test.csv")["label"]

    # Load best model
    model = joblib.load(f"models/{task}_LinearSVM.joblib")

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="weighted", zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, average="weighted", zero_division=0))
    print("Weighted F1:", f1_score(y_test, y_pred, average="weighted", zero_division=0))

    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, zero_division=0)
    print(report)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save predictions
    pd.DataFrame({
        "actual_label": y_test,
        "predicted_label": y_pred
    }).to_csv(f"models/{task}_test_predictions.csv", index=False)

print("\nFinal evaluation completed!")
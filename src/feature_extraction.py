import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz
from joblib import dump

ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = ROOT / "data" / "processed"
OUTPUT_FOLDER = ROOT / "data" / "features"

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

tasks = ["sentiment", "offensive"]

for task in tasks:

    print(f"\nProcessing {task} dataset...")

    train = pd.read_csv(
        DATA_FOLDER / f"tamil_{task}_full_train_processed.csv"
    )

    dev = pd.read_csv(
        DATA_FOLDER / f"tamil_{task}_full_dev_processed.csv"
    )

    test = pd.read_csv(
        DATA_FOLDER / f"tamil_{task}_full_test_processed.csv"
    )

    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2
    )

    # Fit only on training data
    X_train = vectorizer.fit_transform(train["clean_text"])

    # Transform dev and test data
    X_dev = vectorizer.transform(dev["clean_text"])
    X_test = vectorizer.transform(test["clean_text"])

    # Save feature matrices
    save_npz(OUTPUT_FOLDER / f"{task}_X_train.npz", X_train)
    save_npz(OUTPUT_FOLDER / f"{task}_X_dev.npz", X_dev)
    save_npz(OUTPUT_FOLDER / f"{task}_X_test.npz", X_test)

    # Save labels
    train["label"].to_csv(
        OUTPUT_FOLDER / f"{task}_y_train.csv", index=False
    )

    dev["label"].to_csv(
        OUTPUT_FOLDER / f"{task}_y_dev.csv", index=False
    )

    test["label"].to_csv(
        OUTPUT_FOLDER / f"{task}_y_test.csv", index=False
    )

    # Save vectorizer
    dump(vectorizer, OUTPUT_FOLDER / f"{task}_vectorizer.joblib")

    print("Training shape:", X_train.shape)
    print("Development shape:", X_dev.shape)
    print("Testing shape:", X_test.shape)
    print("Features:", len(vectorizer.get_feature_names_out()))

print("\nTF-IDF extraction completed!")
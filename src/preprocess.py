import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_FOLDER = ROOT / "data" / "raw"
PROCESSED_FOLDER = ROOT / "data" / "processed"

PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)


def clean_text(text):
    text = str(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


files = {
    "sentiment": [
        "tamil_sentiment_full_train.csv",
        "tamil_sentiment_full_dev.csv",
        "tamil_sentiment_full_test.csv"
    ],
    "offensive": [
        "tamil_offensive_full_train.csv",
        "tamil_offensive_full_dev.csv",
        "tamil_offensive_full_test.csv"
    ]
}

for task, filenames in files.items():

    for filename in filenames:

        file_path = RAW_FOLDER / filename

        delimiter = ";" if filename in [
            "tamil_sentiment_full_dev.csv",
            "tamil_sentiment_full_test.csv"
        ] else "\t"

        rows = []

        with open(file_path, "r", encoding="utf-8-sig") as f:
            for line in f:

                line = line.rstrip("\r\n").rstrip(delimiter)

                if delimiter not in line:
                    continue

                text, label = line.rsplit(delimiter, 1)

                rows.append({
                    "text": text.strip(),
                    "label": label.strip()
                })

        df = pd.DataFrame(rows)

        # Clean text
        df["clean_text"] = df["text"].apply(clean_text)

        # Remove empty records
        df = df[
            (df["clean_text"] != "") &
            (df["label"] != "")
        ]

        output_name = filename.replace(
            ".csv", "_processed.csv"
        )

        output_path = PROCESSED_FOLDER / output_name

        df.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"\nProcessed: {filename}")
        print("Records:", len(df))
        print("Saved:", output_path)
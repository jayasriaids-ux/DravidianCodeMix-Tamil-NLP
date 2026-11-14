from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
DATA_FOLDER = ROOT / "data" / "raw"

files = [
    "tamil_sentiment_full_train.csv",
    "tamil_sentiment_full_dev.csv",
    "tamil_sentiment_full_test.csv",
    "tamil_offensive_full_train.csv",
    "tamil_offensive_full_dev.csv",
    "tamil_offensive_full_test.csv"
]

for filename in files:
    file_path = DATA_FOLDER / filename

    # Sentiment dev/test use semicolon; other files use tab
    delimiter = ";" if filename in [
        "tamil_sentiment_full_dev.csv",
        "tamil_sentiment_full_test.csv"
    ] else "\t"

    rows = []
    invalid_rows = []

    with open(file_path, "r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\r\n").rstrip(delimiter)

            if not line:
                continue

            if delimiter not in line:
                invalid_rows.append((line_number, line))
                continue

            text, label = line.rsplit(delimiter, 1)
            rows.append((text, label.strip()))

    labels = Counter(label for text, label in rows)

    print("\n" + "=" * 45)
    print(filename)
    print("Total records:", len(rows))
    print("Invalid rows:", len(invalid_rows))
    print("Labels:")

    for label, count in labels.most_common():
        print(repr(label), ":", count)

    if invalid_rows:
        print("Invalid examples:", invalid_rows[:3])
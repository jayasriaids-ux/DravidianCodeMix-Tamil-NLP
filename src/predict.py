import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained models
sentiment_model = joblib.load("models/sentiment_LinearSVM.joblib")
offensive_model = joblib.load("models/offensive_LinearSVM.joblib")

# Load TF-IDF vectorizers
sentiment_vectorizer = joblib.load(
    "data/features/sentiment_vectorizer.joblib"
)

offensive_vectorizer = joblib.load(
    "data/features/offensive_vectorizer.joblib"
)


def clean_text(text):
    text = text.lower()
    text = " ".join(text.split())
    return text


def predict_comment(text):

    cleaned_text = clean_text(text)

    # Sentiment prediction
    sentiment_features = sentiment_vectorizer.transform([cleaned_text])
    sentiment = sentiment_model.predict(sentiment_features)[0]

    # Offensive prediction
    offensive_features = offensive_vectorizer.transform([cleaned_text])
    offensive = offensive_model.predict(offensive_features)[0]

    print("\n========== PREDICTION ==========")
    print("Comment:", text)
    print("Sentiment:", sentiment)
    print("Offensive Category:", offensive)
    print("================================")


while True:

    comment = input("\nEnter a Tamil-English comment (or type exit): ")

    if comment.lower() == "exit":
        print("Prediction program closed.")
        break

    predict_comment(comment)
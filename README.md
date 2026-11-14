# DravidianCodeMix: Tamil Sentiment Analysis and Offensive Language Detection

## 1. Project Overview

This project uses Machine Learning to analyze Tamil and Tamil-English code-mixed text.

It performs two NLP tasks:

- **Sentiment Analysis:** Classifies comments into sentiment categories.
- **Offensive Language Detection:** Identifies offensive comments and their categories.

## 2. Dataset

Dataset: DravidianCodeMix

Research Paper: DravidianCodeMix: Sentiment Analysis and Offensive Language Identification Dataset for Dravidian Languages in Code-Mixed Text (2022).

The project uses Tamil sentiment and offensive language datasets.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Machine Learning
- Joblib

## 4. Machine Learning Algorithms

Three algorithms were trained and compared:

1. Logistic Regression
2. Multinomial Naive Bayes
3. Linear Support Vector Machine (Linear SVM)

Linear SVM was selected for both tasks based on development-set Weighted F1.

## 5. Project Structure

```text
DravidianCodeMix-Project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
│
├── src/
│   ├── inspect_data.py
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── predict.py
│
├── models/
│   ├── sentiment_LinearSVM.joblib
│   ├── offensive_LinearSVM.joblib
│   └── model_comparison.csv
│
└── README.md
```

## 6. Model Performance

Final test results:

| Metric | Sentiment Analysis | Offensive Detection |
|---|---:|---:|
| Accuracy | 60.70% | 76.53% |
| Precision | 57.71% | 72.33% |
| Recall | 60.70% | 76.53% |
| Weighted F1 | 58.54% | 73.56% |

## 7. Sample Predictions

**Example 1**

Input:
`இந்த படம் ரொம்ப நல்லா இருக்கு`

- Sentiment: Positive
- Offensive Category: Not_offensive

**Example 2**

Input:
`nee romba worst ah pesura`

- Sentiment: Negative
- Offensive Category: Offensive_Targeted_Insult_Individual

## 8. How to Run

Install the required packages:

```bash
python -m pip install pandas numpy scipy scikit-learn joblib
```

Run preprocessing:

```bash
python src/preprocess.py
```

Extract features:

```bash
python src/feature_extraction.py
```

Train models:

```bash
python src/train_models.py
```

Evaluate models:

```bash
python src/evaluate_models.py
```

Run predictions:

```bash
python src/predict.py
```

## 9. Future Improvements

- Improve minority-class classification.
- Experiment with advanced NLP techniques.
- Explore deep learning models.
- Improve Tamil-English text understanding.

## 10. Conclusion

This project demonstrates an end-to-end Machine Learning workflow for Tamil and Tamil-English text classification, covering data preparation, feature extraction, model training, evaluation, and prediction.
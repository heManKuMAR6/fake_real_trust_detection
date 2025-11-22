import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk
import warnings
warnings.filterwarnings("ignore")

nltk.download("punkt")

# =========================
# 1. LOAD DATA
# =========================
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Assign labels
true_df["label"] = 1   # Real news
fake_df["label"] = 0   # Fake news

# Combine datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42)  # Shuffle everything

# Prepare features and labels
X = df["text"].astype(str)
y = df["label"]


# =========================
# 2. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. TF-IDF VECTORIZATION
# =========================
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =========================
# 4. TRAIN LOGISTIC REGRESSION MODEL
# =========================
model = LogisticRegression(max_iter=200)
model.fit(X_train_tfidf, y_train)

# =========================
# 5. PREDICT ON TEST DATA
# =========================
y_pred = model.predict(X_test_tfidf)

# =========================
# 6. EVALUATE THE MODEL
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== TF-IDF + Logistic Regression Results ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


input("\nPress Enter to exit...")

import joblib

# Save the vectorizer and model
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(model, "logreg_model.pkl")

print("\nSaved vectorizer as tfidf_vectorizer.pkl")
print("Saved model as logreg_model.pkl")

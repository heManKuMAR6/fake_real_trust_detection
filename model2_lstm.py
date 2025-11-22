import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

import warnings
warnings.filterwarnings("ignore")

# =========================
# 1. LOAD AND PREPARE DATA
# =========================

# Load datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Assign labels: 1 = True, 0 = Fake
true_df["label"] = 1
fake_df["label"] = 0

# Combine and shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42)

# Use only the "text" column
X = df["text"].astype(str).values
y = df["label"].values


# =========================
# 2. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. TOKENIZATION
# =========================
vocab_size = 10000     # limit words (keeps it light)
max_length = 200        # max sequence length

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text → sequence of numbers
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences to equal length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')


# =========================
# 4. BUILD LSTM MODEL
# =========================
embedding_dim = 32   # small embedding size to keep it light

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_length),
    LSTM(32),                      # lightweight LSTM
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # output: fake or real
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()


# =========================
# 5. TRAIN THE MODEL
# =========================
epochs = 2       # light training (you can increase later)
batch_size = 128 # fast + efficient

history = model.fit(
    X_train_pad,
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,
    verbose=1
)

# =========================
# 6. EVALUATE THE MODEL
# =========================
# Predict probabilities
y_pred_prob = model.predict(X_test_pad)

# Convert probabilities → binary labels
y_pred = (y_pred_prob > 0.5).astype("int32")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== LSTM Model Results ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# =========================
# 7. SAVE MODEL + TOKENIZER
# =========================
model.save("lstm_model.h5")

import pickle
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("\nSaved LSTM model as lstm_model.h5")
print("Saved tokenizer as tokenizer.pkl")

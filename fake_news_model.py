# notebooks/fake_news_model.py

import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, "datasets", "Fake_news.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ================= LOAD DATA =================
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH, encoding="latin-1")

# ================= NORMALIZE COLUMNS =================
df.columns = [c.lower().strip() for c in df.columns]

# ================= REQUIRED COLUMNS =================
if "text" not in df.columns or "label" not in df.columns:
    raise Exception("CSV must have 'text' and 'label' columns")

# ================= CLEAN TEXT =================
df["text"] = df["text"].astype(str)

# ================= FIX LABEL =================
# Works for: FAKE/REAL OR 0/1
df["label"] = df["label"].astype(str).str.upper().str.strip()

df["label"] = df["label"].map({
    "FAKE": 1,
    "REAL": 0,
    "1": 1,
    "0": 0
})

df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("✅ Label mapping done")
print(df["label"].value_counts())

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ================= TF-IDF =================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.95,
    min_df=2,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ================= MODEL =================
model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced"
)

model.fit(X_train_vec, y_train)

# ================= ACCURACY =================
preds = model.predict(X_test_vec)
acc = accuracy_score(y_test, preds)

print(f"🎯 Fake News Model Accuracy: {round(acc*100,2)}%")

# ================= SAVE =================
pickle.dump(model, open(os.path.join(MODELS_DIR, "fake_news_model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODELS_DIR, "fake_news_vectorizer.pkl"), "wb"))

print("✅ Fake News Model Trained & Saved Successfully")

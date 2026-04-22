# notebooks/fake_news_model_v2.py
"""
IMPROVED FAKE NEWS DETECTION MODEL - Version 2
===============================================
Simpler ensemble with better accuracy
"""

import pandas as pd
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, "datasets", "Fake_news.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ================= TEXT PREPROCESSING =================
def clean_text(text):
    """Clean text for fake news detection"""
    if not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep letters, numbers, and basic punctuation
    text = re.sub(r'[^a-z0-9\s!?.,]', '', text)
    
    # Multiple spaces to single
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# ================= LOAD DATA =================
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH, encoding="latin-1")

# ================= NORMALIZE COLUMNS =================
df.columns = [c.lower().strip() for c in df.columns]

# Check columns
print(f"Columns found: {df.columns.tolist()}")

if "text" not in df.columns or "label" not in df.columns:
    raise Exception("CSV must have 'text' and 'label' columns")

# ================= CLEAN TEXT =================
print("🧹 Cleaning text...")
df["cleaned_text"] = df["text"].apply(clean_text)

# ================= FIX LABEL =================
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

# ================= SPLIT DATA =================
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# ================= TF-IDF VECTORIZATION =================
print("\n🔧 Building TF-IDF features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.85,
    min_df=3,
    ngram_range=(1, 2),  # Unigrams and bigrams
    sublinear_tf=True,
    use_idf=True,
    smooth_idf=True,
    norm='l2',
    max_features=10000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✓ Feature matrix shape: {X_train_vec.shape}")

# ================= TRAIN ENSEMBLE =================
print("\n🚀 Training Models...\n")

# Model 1: Logistic Regression
print("1️⃣ Training Logistic Regression...")
lr = LogisticRegression(
    max_iter=3000,
    C=0.5,
    class_weight="balanced",
    solver='lbfgs',
    random_state=42
)
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"   Accuracy: {round(lr_acc*100, 2)}%")

# Model 2: Naive Bayes (Great for text)
print("2️⃣ Training Naive Bayes...")
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_vec, y_train)
nb_pred = nb.predict(X_test_vec)
nb_acc = accuracy_score(y_test, nb_pred)
print(f"   Accuracy: {round(nb_acc*100, 2)}%")

# ================= ENSEMBLE PREDICTION =================
print("\n🎯 Creating Ensemble...")

# Get probabilities
lr_probs = lr.predict_proba(X_test_vec)
nb_probs = nb.predict_proba(X_test_vec)

# Average probabilities
ensemble_probs = (lr_probs + nb_probs) / 2
ensemble_pred = np.argmax(ensemble_probs, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n🏆 Ensemble Accuracy: {round(ensemble_acc*100, 2)}%")

# ================= EVALUATION =================
print("\n📊 Detailed Evaluation:")
print("="*60)
print(classification_report(y_test, ensemble_pred, target_names=['REAL', 'FAKE']))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, ensemble_pred)
print(cm)
print("="*60)

# Calculate precision and recall for each class
tn, fp, fn, tp = cm.ravel()
print(f"\nREAL News Detection:")
print(f"  True Positives: {tp}")
print(f"  False Positives: {fp}")
print(f"  Precision: {round(tp/(tp+fp)*100, 2)}%")
print(f"  Recall: {round(tp/(tp+fn)*100, 2)}%")

print(f"\nFAKE News Detection:")
real_tn = tn
real_fp = fn
real_fn = fp
real_tp = tp
print(f"  True Negatives (detected as REAL): {real_tn}")
print(f"  Accuracy: {round((real_tn+real_tp)/(tp+tn+fp+fn)*100, 2)}%")

# ================= SAVE MODELS =================
print("\n💾 Saving models...")

model_data = {
    'lr_model': lr,
    'nb_model': nb,
    'accuracy': ensemble_acc
}

pickle.dump(model_data, open(os.path.join(MODELS_DIR, "fake_news_model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODELS_DIR, "fake_news_vectorizer.pkl"), "wb"))

print(f"\n✅ Fake News Model Trained & Saved!")
print(f"🎯 Final Accuracy: {round(ensemble_acc*100, 2)}%")
print(f"📦 Saved to: {MODELS_DIR}")

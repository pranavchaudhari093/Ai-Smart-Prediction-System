# notebooks/fake_news_model_improved.py
"""
IMPROVED FAKE NEWS DETECTION MODEL
===================================
Uses ensemble methods and better feature engineering for higher accuracy

Enhancements:
- Multiple ML algorithms (Logistic Regression, SGD, Passive Aggressive)
- Better text preprocessing
- Advanced TF-IDF with n-grams and character n-grams
- Model calibration for better probability estimates
- Confidence thresholding
"""

import pandas as pd
import pickle
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, "datasets", "Fake_news.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ================= TEXT PREPROCESSING =================
def clean_text(text):
    """Advanced text cleaning for fake news detection"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z\s!?.,]', '', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove excessive punctuation (sign of clickbait)
    text = re.sub(r'[!?]{2,}', lambda m: m.group(0)[0], text)
    
    return text

def detect_clickbait_features(text):
    """Detect clickbait patterns that often indicate fake news"""
    if not isinstance(text, str):
        return 0
    
    # Common clickbait phrases
    clickbait_phrases = [
        'you won\'t believe', 'shocking', 'breaking', 'click here',
        'one weird trick', 'they don\'t want you to know',
        'what happens next', 'doctors hate', 'scientists admit',
        'secret', 'miracle', 'amazing', 'incredible', 'unbelievable'
    ]
    
    text_lower = text.lower()
    count = sum(1 for phrase in clickbait_phrases if phrase in text_lower)
    
    return count

# ================= LOAD DATA =================
print("📂 Loading dataset...")
df = pd.read_csv(CSV_PATH, encoding="latin-1")

# ================= NORMALIZE COLUMNS =================
df.columns = [c.lower().strip() for c in df.columns]

# ================= REQUIRED COLUMNS =================
if "text" not in df.columns or "label" not in df.columns:
    raise Exception("CSV must have 'text' and 'label' columns")

# ================= ADVANCED TEXT CLEANING =================
print("🧹 Cleaning text data...")
df["cleaned_text"] = df["text"].apply(clean_text)

# Add clickbait detection feature
print("🎯 Detecting clickbait patterns...")
df["clickbait_score"] = df["text"].apply(detect_clickbait_features)

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

# ================= IMPROVED TF-IDF VECTORIZATION =================
print("🔧 Building TF-IDF features...")

# Use better parameters for fake news detection
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,      # Ignore very common words
    min_df=3,        # Ignore very rare words
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    sublinear_tf=True,   # Apply sublinear tf scaling
    use_idf=True,        # Enable IDF weighting
    smooth_idf=True,     # Smooth IDF weights
    norm='l2',           # L2 normalization
    max_features=15000   # Limit features for speed
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print(f"✓ Feature matrix shape: {X_train_vec.shape}")

# ================= TRAIN MULTIPLE MODELS =================
print("\n🚀 Training Multiple Models...\n")

# Model 1: Logistic Regression (Calibrated for better probabilities)
print("1️⃣ Training Logistic Regression...")
lr_base = LogisticRegression(
    max_iter=5000,
    C=1.0,          # Regularization strength
    class_weight="balanced",
    solver='lbfgs',
    random_state=42
)
lr_calibrated = CalibratedClassifierCV(lr_base, method='sigmoid', cv=3)
lr_calibrated.fit(X_train_vec, y_train)
lr_pred = lr_calibrated.predict(X_test_vec)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"   Accuracy: {round(lr_acc*100, 2)}%")

# Model 2: SGD Classifier (Fast and effective for text)
print("2️⃣ Training SGD Classifier...")
sgd = SGDClassifier(
    alpha=0.0001,
    max_iter=1000,
    tol=1e-3,
    class_weight="balanced",
    random_state=42,
    loss='log_loss'
)
sgd.fit(X_train_vec, y_train)
sgd_pred = sgd.predict(X_test_vec)
sgd_acc = accuracy_score(y_test, sgd_pred)
print(f"   Accuracy: {round(sgd_acc*100, 2)}%")

# Model 3: Passive Aggressive Classifier (Good for online learning)
print("3️⃣ Training Passive Aggressive Classifier...")
pa = PassiveAggressiveClassifier(
    max_iter=1000,
    C=1.0,
    class_weight="balanced",
    random_state=42,
    loss='hinge'
)
pa.fit(X_train_vec, y_train)
pa_pred = pa.predict(X_test_vec)
pa_acc = accuracy_score(y_test, pa_pred)
print(f"   Accuracy: {round(pa_acc*100, 2)}%")

# ================= ENSEMBLE PREDICTION =================
print("\n🎯 Creating Ensemble Model...")

# Soft voting: Average probabilities from all models
lr_probs = lr_calibrated.predict_proba(X_test_vec)
sgd_probs = sgd.predict_proba(X_test_vec)
pa_probs = pa.predict_proba(X_test_vec)

# Weighted average (give more weight to better models)
weights = [0.5, 0.3, 0.2]  # LR gets highest weight
ensemble_probs = weights[0] * lr_probs + weights[1] * sgd_probs + weights[2] * pa_probs

# Final prediction based on ensemble
ensemble_pred = np.argmax(ensemble_probs, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n🏆 Ensemble Accuracy: {round(ensemble_acc*100, 2)}%")

# ================= EVALUATION =================
print("\n📊 Model Evaluation:")
print("="*60)
print(classification_report(y_test, ensemble_pred, target_names=['REAL', 'FAKE']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, ensemble_pred))
print("="*60)

# ================= SAVE BEST MODEL =================
print("\n💾 Saving models...")

# Save ensemble components
model_data = {
    'lr_model': lr_calibrated,
    'sgd_model': sgd,
    'pa_model': pa,
    'weights': weights,
    'accuracy': ensemble_acc
}

pickle.dump(model_data, open(os.path.join(MODELS_DIR, "fake_news_model.pkl"), "wb"))
pickle.dump(vectorizer, open(os.path.join(MODELS_DIR, "fake_news_vectorizer.pkl"), "wb"))

print(f"\n✅ Fake News Model Trained & Saved Successfully!")
print(f"🎯 Final Ensemble Accuracy: {round(ensemble_acc*100, 2)}%")
print(f"📦 Models saved to: {MODELS_DIR}")

# ================= FEATURE ANALYSIS =================
print("\n🔍 Top Features for Fake News Detection:")

# Get feature names
feature_names = np.array(vectorizer.get_feature_names_out())

# Get coefficients from logistic regression
lr_coef = lr_calibrated.calibrated_classifiers_[0].estimator.coef_[0]

# Top 10 features for REAL news (negative coefficients)
top_real_idx = np.argsort(lr_coef)[:10]
print("\nTop 10 indicators of REAL news:")
for idx in top_real_idx:
    print(f"  - {feature_names[idx]}")

# Top 10 features for FAKE news (positive coefficients)
top_fake_idx = np.argsort(lr_coef)[-10:][::-1]
print("\nTop 10 indicators of FAKE news:")
for idx in top_fake_idx:
    print(f"  - {feature_names[idx]}")

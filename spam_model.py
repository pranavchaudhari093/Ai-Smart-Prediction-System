import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ================= PATH FIX (IMPORTANT) =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(BASE_DIR, "datasets", "spam.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "spam_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "spam_vectorizer.pkl")

# ================= LOAD DATA =================
df = pd.read_csv(DATASET_PATH, encoding="latin-1")

# Rename columns if dataset is standard spam dataset
df = df.iloc[:, :2]
df.columns = ["label", "message"]

# Convert label to binary
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# ================= SPLIT =================
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= VECTORIZATION =================
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ================= MODEL =================
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ================= EVALUATION =================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Spam Model Accuracy:", round(accuracy * 100, 2), "%")

# ================= SAVE =================
pickle.dump(model, open(MODEL_PATH, "wb"))
pickle.dump(vectorizer, open(VECTORIZER_PATH, "wb"))

print("✅ Model & Vectorizer saved successfully")

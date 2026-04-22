import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("📂 Loading Loan Dataset...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, "datasets", "loan_data.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

TARGET = "loan_status"

categorical_cols = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file"
]

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).str.lower())
    encoders[col] = le

X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 🔥 BETTER MODEL
model = LogisticRegression(
    max_iter=6000,
    class_weight="balanced",
    solver="lbfgs"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100

print(f"✅ Model Accuracy: {accuracy:.2f}%")

# SAVE EVERYTHING
pickle.dump(model, open(os.path.join(MODELS_DIR, "loan_model.pkl"), "wb"))
pickle.dump(encoders, open(os.path.join(MODELS_DIR, "loan_encoders.pkl"), "wb"))
pickle.dump(round(accuracy, 2), open(os.path.join(MODELS_DIR, "loan_accuracy.pkl"), "wb"))

print("✅ Loan Model + Accuracy Saved Successfully")

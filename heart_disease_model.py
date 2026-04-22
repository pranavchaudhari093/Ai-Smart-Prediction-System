import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ================= CORRECT PATH =================
NOTEBOOKS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(NOTEBOOKS_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, "datasets", "heart_disease_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")

# ================= CHECK CSV =================
if not os.path.exists(CSV_PATH):
    print("❌ CSV NOT FOUND AT:", CSV_PATH)
    exit()

os.makedirs(MODELS_DIR, exist_ok=True)

# ================= LOAD DATA =================
print("✅ Loading CSV from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# ================= HANDLE NULL =================
df["Alcohol Intake"] = df["Alcohol Intake"].fillna("None")

# ================= ENCODING =================
encoders = {}
categorical_cols = [
    "Gender",
    "Smoking",
    "Alcohol Intake",
    "Family History",
    "Diabetes",
    "Obesity",
    "Exercise Induced Angina",
    "Chest Pain Type"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ================= FEATURES & TARGET =================
X = df[
    [
        "Age",
        "Gender",
        "Cholesterol",
        "Blood Pressure",
        "Heart Rate",
        "Smoking",
        "Alcohol Intake",
        "Exercise Hours",
        "Family History",
        "Diabetes",
        "Obesity",
        "Stress Level",
        "Blood Sugar",
        "Exercise Induced Angina",
        "Chest Pain Type",
    ]
]

y = df["Heart Disease"]

# ================= TRAIN MODEL =================
model = LogisticRegression(max_iter=3000)
model.fit(X, y)

# ================= SAVE =================
pickle.dump(model, open(os.path.join(MODELS_DIR, "heart_model.pkl"), "wb"))
pickle.dump(encoders, open(os.path.join(MODELS_DIR, "heart_encoders.pkl"), "wb"))

print("✅ Heart Disease Model Trained Successfully")

# notebooks/loan_model_improved.py
"""
IMPROVED LOAN APPROVAL PREDICTION MODEL
========================================
Better feature engineering and model calibration for accurate predictions
"""

import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

print("📂 Loading Loan Dataset...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

CSV_PATH = os.path.join(PROJECT_DIR, "datasets", "loan_data.csv")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)

print(f"Dataset shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['loan_status'].value_counts())

# Define target
TARGET = "loan_status"

# Categorical columns to encode
categorical_cols = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file"
]

# Encode categorical variables
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str).str.lower())
    encoders[col] = le
    print(f"\n{col} encoding:")
    print(f"  Classes: {le.classes_}")

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

print(f"\nFeature columns: {X.columns.tolist()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale numerical features (important for logistic regression)
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("\n✅ Feature scaling complete")

# Train multiple models
print("\n🚀 Training Models...\n")

# Model 1: Logistic Regression with better parameters
print("1️⃣ Training Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=5000,
    C=0.5,  # Regularization
    class_weight="balanced",
    solver='lbfgs',
    random_state=42
)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"   Accuracy: {round(lr_acc*100, 2)}%")

# Model 2: Random Forest
print("2️⃣ Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"   Accuracy: {round(rf_acc*100, 2)}%")

# Model 3: Gradient Boosting
print("3️⃣ Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
print(f"   Accuracy: {round(gb_acc*100, 2)}%")

# Ensemble: Average probabilities from all models
print("\n🎯 Creating Ensemble Prediction...")
lr_probs = lr_model.predict_proba(X_test_scaled)
rf_probs = rf_model.predict_proba(X_test_scaled)
gb_probs = gb_model.predict_proba(X_test_scaled)

# Weighted average (RF and GB usually perform better)
weights = [0.3, 0.35, 0.35]
ensemble_probs = weights[0] * lr_probs + weights[1] * rf_probs + weights[2] * gb_probs
ensemble_pred = np.argmax(ensemble_probs, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print(f"\n🏆 Ensemble Accuracy: {round(ensemble_acc*100, 2)}%")

# Detailed evaluation
print("\n📊 Detailed Evaluation:")
print("="*60)
print(classification_report(y_test, ensemble_pred, target_names=['Not Approved', 'Approved']))
print("Confusion Matrix:")
cm = confusion_matrix(y_test, ensemble_pred)
print(cm)
print("="*60)

# Analyze which class is which
tn, fp, fn, tp = cm.ravel()
print(f"\nClass 0 (Not Approved):")
print(f"  True Negatives: {tn}")
print(f"  False Positives: {fp}")
print(f"  Precision: {round(tn/(tn+fn)*100, 2)}%")

print(f"\nClass 1 (Approved):")
print(f"  True Positives: {tp}")
print(f"  False Negatives: {fn}")
print(f"  Precision: {round(tp/(tp+fp)*100, 2)}%")

# Feature importance (from Random Forest)
print("\n🔍 Feature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(feature_importance)

# Save models
print("\n💾 Saving models...")

model_data = {
    'lr_model': lr_model,
    'rf_model': rf_model,
    'gb_model': gb_model,
    'encoders': encoders,
    'scaler': scaler,
    'weights': weights,
    'accuracy': ensemble_acc,
    'feature_columns': X.columns.tolist()
}

pickle.dump(model_data, open(os.path.join(MODELS_DIR, "loan_model.pkl"), "wb"))
pickle.dump(round(ensemble_acc, 2), open(os.path.join(MODELS_DIR, "loan_accuracy.pkl"), "wb"))

print(f"\n✅ Loan Model Trained & Saved Successfully!")
print(f"🎯 Final Ensemble Accuracy: {round(ensemble_acc*100, 2)}%")
print(f"📦 Models saved to: {MODELS_DIR}")

# Test with sample data
print("\n🧪 Testing with sample data...")

# Create sample input
sample_input = pd.DataFrame([{
    'person_age': 25,
    'person_income': 50000,
    'person_emp_length': 3,
    'person_gender': 1,  # male
    'person_education': 2,  # bachelor
    'person_home_ownership': 2,  # own
    'loan_amnt': 10000,
    'loan_intent': 0,  # personal
    'loan_int_rate': 10.5,
    'loan_percent_income': 0.2,
    'cred_hist_length': 5,
    'person_credit_score': 750,
    'previous_loan_defaults_on_file': 0  # no default
}])

# Scale the sample
sample_scaled = sample_input.copy()
sample_scaled[numerical_cols] = scaler.transform(sample_input[numerical_cols])

# Predict
sample_probs = lr_model.predict_proba(sample_scaled)[0]
print(f"\nSample prediction probabilities:")
print(f"  Class 0 (Not Approved): {sample_probs[0]*100:.2f}%")
print(f"  Class 1 (Approved): {sample_probs[1]*100:.2f}%")
print(f"  Prediction: {'APPROVED' if sample_probs[1] > sample_probs[0] else 'REJECTED'}")

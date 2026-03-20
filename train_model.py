"""
Run this script ONCE locally to generate heart_disease_rf_model.pkl
Then commit that .pkl file to your GitHub repo along with app.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv("heart_disease_cleaned.csv")

df = df.replace({
    "Yes": 1, "No": 0,
    "yes": 1, "no": 0,
    "Male": 1, "Female": 0,
    "male": 1, "female": 0,
    "Presence": 1, "Absence": 0,
    "presence": 1, "absence": 0
})

for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].astype(str)

le = LabelEncoder()
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = le.fit_transform(df[col])

X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1       # uses all CPU cores — trains faster
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Accuracy: {acc:.4f}")

joblib.dump(model, "heart_disease_rf_model.pkl")
print("✅ Model saved as heart_disease_rf_model.pkl — commit this file to your repo!")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from groq import Groq
import warnings
warnings.filterwarnings('ignore')

# ✅ MODEL CONFIG (FIXED)
MODEL_NAME = "llama3-70b-8192"

# PAGE CONFIG
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# LOAD MODEL
@st.cache_resource
def load_model():
    df = pd.read_csv("cleaned_heart.csv")

    for col in ['AgeGroup', 'BP_Category']:
        if col in df.columns:
            df = df.drop(columns=[col])

    cat_features = ['Gender', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    encoders = {}

    for col in cat_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model, encoders

model, encoders = load_model()

# AI FUNCTION (FIXED)
def get_ai_recommendation(patient_name, age, gender, chest_pain, bp,
                         cholesterol, max_hr, ex_angina, oldpeak,
                         st_slope, fasting_bs, prediction, probability):

    try:
        api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=api_key)
    except Exception:
        raise ValueError("GROQ_API_KEY missing in Streamlit secrets.")

    risk_level = "HIGH RISK — Heart Disease Detected" if prediction == 1 else "LOW RISK — No Heart Disease"
    disease_prob = f"{probability[1] * 100:.1f}%"

    prompt = f"""
Patient: {patient_name}, Age: {age}, Gender: {gender}
Chest Pain: {chest_pain}, BP: {bp}, Cholesterol: {cholesterol}
Max HR: {max_hr}, Angina: {ex_angina}, Oldpeak: {oldpeak}
ST Slope: {st_slope}, Fasting BS: {fasting_bs}

Result: {risk_level}
Probability: {disease_prob}

Give:
- Short greeting
- Explanation
- 5 health tips
- Lifestyle changes
- Doctor advice
End with disclaimer.
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,  # ✅ FIXED
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        return "⚠️ AI recommendations temporarily unavailable. Please consult a doctor."

# UI
st.title("❤️ Heart Disease Prediction")

patient_name = st.text_input("Patient Name")

age = st.slider("Age", 18, 100, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
chest = st.selectbox("Chest Pain", ["ASY", "ATA", "NAP", "TA"])
bp = st.number_input("Resting BP", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
max_hr = st.slider("Max HR", 60, 200, 140)
angina = st.selectbox("Exercise Angina", ["Yes", "No"])
oldpeak = st.number_input("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["UP", "FLAT", "DOWN"])
fasting = st.selectbox("Fasting BS", ["Yes", "No"])

if st.button("Predict"):

    if not patient_name:
        st.warning("Enter patient name")
        st.stop()

    input_data = pd.DataFrame([{
        'Age': age,
        'Gender': encoders['Gender']['M'] if gender == "Male" else encoders['Gender']['F'],
        'ChestPainType': encoders['ChestPainType'][chest],
        'RestingBloodPressure': bp,
        'Cholesterol': chol,
        'FastingBloodSugar': 1 if fasting == "Yes" else 0,
        'RestingECG': encoders['RestingECG']['NORMAL'],
        'MaxHR': max_hr,
        'ExerciseAngina': encoders['ExerciseAngina']['Y'] if angina == "Yes" else encoders['ExerciseAngina']['N'],
        'Oldpeak': oldpeak,
        'ST_Slope': encoders['ST_Slope'][st_slope]
    }])

    pred = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    if pred == 1:
        st.error(f"HIGH RISK ({proba[1]*100:.1f}%)")
    else:
        st.success(f"LOW RISK ({proba[0]*100:.1f}%)")

    st.subheader("🤖 AI Recommendation")

    ai_text = get_ai_recommendation(
        patient_name, age, gender, chest,
        bp, chol, max_hr, angina,
        oldpeak, st_slope, fasting,
        pred, proba
    )

    st.write(ai_text)

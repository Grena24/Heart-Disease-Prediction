import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

AGE_CATEGORIES = [
    "18-24", "25-29", "30-34", "35-39", "40-44",
    "45-49", "50-54", "55-59", "60-64", "65-69",
    "70-74", "75-79", "80 or older"
]
ETHNICITIES = sorted([
    "American Indian/Alaskan Native", "Asian", "Black",
    "Hispanic", "Other", "White"
])
GENERAL_HEALTH = ["Poor", "Fair", "Good", "Very good", "Excellent"]

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_rf_model.pkl")

def encode_inputs(bmi, smoking, alcohol, stroke, physical_health,
                  mental_health, walking_diff, gender, age_cat,
                  ethnicity, diabetic, physical_activity,
                  general_health, sleep_time, kidney_disease):
    age_map = {cat: i for i, cat in enumerate(AGE_CATEGORIES)}
    ethnicity_map = {e: i for i, e in enumerate(ETHNICITIES)}
    health_map = {h: i for i, h in enumerate(GENERAL_HEALTH)}

    return pd.DataFrame([{
        "BMI": bmi,
        "Smoking": int(smoking),
        "AlcoholDrinking": int(alcohol),
        "Stroke": int(stroke),
        "PhysicalHealth": physical_health,
        "MentalHealth": mental_health,
        "WalkingDifficulty": int(walking_diff),
        "Gender": 1 if gender == "Male" else 0,
        "AgeCategory": age_map.get(age_cat, 7),
        "Ethnicity": ethnicity_map.get(ethnicity, 5),
        "Diabetic": int(diabetic),
        "PhysicalActivity": int(physical_activity),
        "GeneralHealth": health_map.get(general_health, 3),
        "SleepTime": sleep_time,
        "KidneyDisease": int(kidney_disease),
    }])

# ── UI ───────────────────────────────────────────────────────────────────────

st.title("🫀 Heart Disease Risk Predictor")
st.markdown("Fill in the details below to estimate the likelihood of heart disease.")

model = load_model()

with st.form("prediction_form"):
    st.subheader("Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age_cat = st.selectbox("Age Category", AGE_CATEGORIES, index=7)
        ethnicity = st.selectbox("Ethnicity", ETHNICITIES, index=5)

    with col2:
        bmi = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        sleep_time = st.number_input("Average Sleep (hours/night)", min_value=1.0, max_value=24.0, value=7.0, step=0.5)
        general_health = st.selectbox("General Health", GENERAL_HEALTH, index=3)

    with col3:
        physical_health = st.slider("Poor Physical Health Days (last 30)", 0, 30, 0)
        mental_health = st.slider("Poor Mental Health Days (last 30)", 0, 30, 0)

    st.subheader("Medical History")
    col4, col5, col6 = st.columns(3)

    with col4:
        smoking = st.checkbox("Smoker (100+ cigarettes lifetime)")
        alcohol = st.checkbox("Heavy Alcohol Drinker")
        stroke = st.checkbox("Had a Stroke")

    with col5:
        diabetic = st.checkbox("Diabetic")
        kidney_disease = st.checkbox("Kidney Disease")
        walking_diff = st.checkbox("Difficulty Walking / Climbing Stairs")

    with col6:
        physical_activity = st.checkbox("Physically Active (last 30 days)", value=True)

    submitted = st.form_submit_button("Predict Risk", use_container_width=True)

if submitted:
    input_df = encode_inputs(
        bmi, smoking, alcohol, stroke, physical_health,
        mental_health, walking_diff, gender, age_cat,
        ethnicity, diabetic, physical_activity,
        general_health, sleep_time, kidney_disease
    )

    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    st.divider()
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        if prediction == 1:
            st.error(f"### ⚠️ Higher Risk Detected\nEstimated probability: **{prob*100:.1f}%**")
        else:
            st.success(f"### ✅ Lower Risk Detected\nEstimated probability: **{prob*100:.1f}%**")

    with col_r2:
        st.metric("Risk Score", f"{prob*100:.1f}%")
        st.progress(float(prob))

    st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")

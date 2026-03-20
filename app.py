import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️", layout="centered")

# Load model
model = joblib.load("heart_disease_rf_model.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Fill in the details below to predict heart disease risk.")

# Mappings
yes_no_map = {"No": 0, "Yes": 1}
gender_map = {"Female": 0, "Male": 1}

age_map = {
    "18-24": 0,
    "25-29": 1,
    "30-34": 2,
    "35-39": 3,
    "40-44": 4,
    "45-49": 5,
    "50-54": 6,
    "55-59": 7,
    "60-64": 8,
    "65-69": 9,
    "70-74": 10,
    "75-79": 11,
    "80 or older": 12
}

ethnicity_map = {
    "American Indian/Alaskan Native": 0,
    "Asian": 1,
    "Black": 2,
    "Hispanic": 3,
    "Other": 4,
    "White": 5
}

diabetic_map = {
    "No": 0,
    "No, borderline diabetes": 1,
    "Yes": 2,
    "Yes (during pregnancy)": 3
}

general_health_map = {
    "Poor": 0,
    "Fair": 1,
    "Good": 2,
    "Very good": 3,
    "Excellent": 4
}

# Inputs
BMI = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
Smoking = yes_no_map[st.selectbox("Smoking", list(yes_no_map.keys()))]
AlcoholDrinking = yes_no_map[st.selectbox("Alcohol Drinking", list(yes_no_map.keys()))]
Stroke = yes_no_map[st.selectbox("Stroke", list(yes_no_map.keys()))]
PhysicalHealth = st.number_input("Physical Health (days in last 30 days)", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
MentalHealth = st.number_input("Mental Health (days in last 30 days)", min_value=0.0, max_value=30.0, value=0.0, step=1.0)
WalkingDifficulty = yes_no_map[st.selectbox("Walking Difficulty", list(yes_no_map.keys()))]
Gender = gender_map[st.selectbox("Gender", list(gender_map.keys()))]
AgeCategory = age_map[st.selectbox("Age Category", list(age_map.keys()))]
Ethnicity = ethnicity_map[st.selectbox("Ethnicity", list(ethnicity_map.keys()))]
Diabetic = diabetic_map[st.selectbox("Diabetic", list(diabetic_map.keys()))]
PhysicalActivity = yes_no_map[st.selectbox("Physical Activity", list(yes_no_map.keys()))]
GeneralHealth = general_health_map[st.selectbox("General Health", list(general_health_map.keys()))]
SleepTime = st.number_input("Sleep Time (hours)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)
KidneyDisease = yes_no_map[st.selectbox("Kidney Disease", list(yes_no_map.keys()))]

# DataFrame
input_data = pd.DataFrame([{
    "BMI": BMI,
    "Smoking": Smoking,
    "AlcoholDrinking": AlcoholDrinking,
    "Stroke": Stroke,
    "PhysicalHealth": PhysicalHealth,
    "MentalHealth": MentalHealth,
    "WalkingDifficulty": WalkingDifficulty,
    "Gender": Gender,
    "AgeCategory": AgeCategory,
    "Ethnicity": Ethnicity,
    "Diabetic": Diabetic,
    "PhysicalActivity": PhysicalActivity,
    "GeneralHealth": GeneralHealth,
    "SleepTime": SleepTime,
    "KidneyDisease": KidneyDisease
}])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    st.dataframe(input_data)

    if prediction == 1:
        st.error("Heart Disease Present")
    else:
        st.success("No Heart Disease")

    st.write(f"Risk Probability: {probability:.4f}")

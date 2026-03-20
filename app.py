import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .patient-header {
        background: linear-gradient(135deg, #e63946, #c1121f);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .patient-header h2 { margin: 0; font-size: 26px; }
    .patient-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 14px; }
    .risk-high {
        background: #fff0f0;
        border-left: 5px solid #e63946;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .risk-low {
        background: #f0fff4;
        border-left: 5px solid #2d6a4f;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
    }
    .section-title {
        font-size: 15px;
        font-weight: 600;
        color: #495057;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

AGE_CATEGORIES = [
    "18-24","25-29","30-34","35-39","40-44",
    "45-49","50-54","55-59","60-64","65-69",
    "70-74","75-79","80 or older"
]
ETHNICITIES = sorted([
    "American Indian/Alaskan Native","Asian","Black",
    "Hispanic","Other","White"
])
GENERAL_HEALTH = ["Poor","Fair","Good","Very good","Excellent"]

RISK_FACTORS = {
    "Smoking": "Smoking",
    "AlcoholDrinking": "Alcohol",
    "Stroke": "Stroke",
    "Diabetic": "Diabetic",
    "KidneyDisease": "Kidney Disease",
    "WalkingDifficulty": "Walking Difficulty",
}

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_rf_model.pkl")

def encode_inputs(bmi, smoking, alcohol, stroke, physical_health,
                  mental_health, walking_diff, gender, age_cat,
                  ethnicity, diabetic, physical_activity,
                  general_health, sleep_time, kidney_disease):
    age_map      = {cat: i for i, cat in enumerate(AGE_CATEGORIES)}
    ethnicity_map = {e: i for i, e in enumerate(ETHNICITIES)}
    health_map   = {h: i for i, h in enumerate(GENERAL_HEALTH)}
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

def gauge_chart(prob):
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    # Background arc
    theta = np.linspace(np.pi, 0, 200)
    for i in range(len(theta) - 1):
        t = i / (len(theta) - 1)
        color = (
            0.18 + 0.72 * t,          # R
            0.60 - 0.47 * t,          # G
            0.18                       # B
        )
        ax.plot([np.cos(theta[i]), np.cos(theta[i+1])],
                [np.sin(theta[i]), np.sin(theta[i+1])],
                color=color, linewidth=14, solid_capstyle="butt")

    # Needle
    angle = np.pi - prob * np.pi
    ax.annotate("", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#1a1a2e",
                                lw=2.5, mutation_scale=18))
    ax.add_patch(plt.Circle((0, 0), 0.08, color="#1a1a2e", zorder=5))

    ax.text(0, -0.22, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#1a1a2e")
    ax.text(-1.0, -0.25, "Low", fontsize=9, color="#2d6a4f", fontweight="600")
    ax.text( 0.80, -0.25, "High", fontsize=9, color="#e63946", fontweight="600")

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.4, 1.1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig

def risk_factor_bar(input_row):
    factors = {
        "Smoking":          input_row["Smoking"].values[0],
        "Alcohol":          input_row["AlcoholDrinking"].values[0],
        "Stroke":           input_row["Stroke"].values[0],
        "Diabetic":         input_row["Diabetic"].values[0],
        "Kidney Disease":   input_row["KidneyDisease"].values[0],
        "Walking Diff.":    input_row["WalkingDifficulty"].values[0],
        "Physical Activity":input_row["PhysicalActivity"].values[0],
    }
    labels = list(factors.keys())
    values = [int(v) for v in factors.values()]
    colors = ["#e63946" if v == 1 else "#adb5bd" for v in values]

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="none")
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top","right","bottom"]].set_visible(False)
    ax.spines["left"].set_color("#dee2e6")
    red_patch   = mpatches.Patch(color="#e63946", label="Present")
    gray_patch  = mpatches.Patch(color="#adb5bd", label="Absent")
    ax.legend(handles=[red_patch, gray_patch], fontsize=8,
              loc="lower right", framealpha=0)
    plt.tight_layout(pad=0.5)
    return fig

def health_radar(physical_health, mental_health, bmi, sleep_time, general_health_idx):
    fig, ax = plt.subplots(figsize=(4, 4),
                           subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    categories = ["Physical\nHealth", "Mental\nHealth", "BMI\nRisk", "Sleep\nQuality", "Gen.\nHealth"]
    # Normalise to 0-1 (1 = worst for health)
    values = [
        physical_health / 30,
        mental_health / 30,
        min(max((bmi - 18.5) / 21.5, 0), 1),
        1 - min(sleep_time / 9, 1),
        1 - general_health_idx / 4,
    ]
    values += values[:1]  # close polygon

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], fontsize=7)
    ax.grid(color="#dee2e6", linewidth=0.8)

    ax.plot(angles, values, color="#e63946", linewidth=2)
    ax.fill(angles, values, color="#e63946", alpha=0.25)
    plt.tight_layout(pad=0.5)
    return fig

# ── App ──────────────────────────────────────────────────────────────────────

st.title("🫀 Heart Disease Risk Predictor")

model = load_model()

with st.form("prediction_form"):

    # Patient name
    st.subheader("Patient Details")
    patient_name = st.text_input("Patient Name", placeholder="e.g. Rahul Sharma")

    st.divider()
    st.subheader("Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender      = st.selectbox("Gender", ["Male", "Female"])
        age_cat     = st.selectbox("Age Category", AGE_CATEGORIES, index=7)
        ethnicity   = st.selectbox("Ethnicity", ETHNICITIES, index=5)

    with col2:
        bmi         = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        sleep_time  = st.number_input("Avg Sleep (hrs/night)", min_value=1.0, max_value=24.0, value=7.0, step=0.5)
        general_health = st.selectbox("General Health", GENERAL_HEALTH, index=3)

    with col3:
        physical_health = st.slider("Poor Physical Health Days (last 30)", 0, 30, 0)
        mental_health   = st.slider("Poor Mental Health Days (last 30)", 0, 30, 0)

    st.subheader("Medical History")
    col4, col5, col6 = st.columns(3)

    with col4:
        smoking     = st.checkbox("Smoker (100+ cigarettes lifetime)")
        alcohol     = st.checkbox("Heavy Alcohol Drinker")
        stroke      = st.checkbox("Had a Stroke")

    with col5:
        diabetic         = st.checkbox("Diabetic")
        kidney_disease   = st.checkbox("Kidney Disease")
        walking_diff     = st.checkbox("Difficulty Walking")

    with col6:
        physical_activity = st.checkbox("Physically Active (last 30 days)", value=True)

    submitted = st.form_submit_button("🔍 Analyse Risk", use_container_width=True)

# ── Results ──────────────────────────────────────────────────────────────────

if submitted:
    name_display = patient_name.strip() if patient_name.strip() else "Patient"

    input_df = encode_inputs(
        bmi, smoking, alcohol, stroke, physical_health,
        mental_health, walking_diff, gender, age_cat,
        ethnicity, diabetic, physical_activity,
        general_health, sleep_time, kidney_disease
    )

    prediction = model.predict(input_df)[0]
    prob       = model.predict_proba(input_df)[0][1]
    gh_idx     = GENERAL_HEALTH.index(general_health)

    # Patient header banner
    st.markdown(f"""
    <div class="patient-header">
        <h2>📋 Report for {name_display}</h2>
        <p>{gender} &nbsp;|&nbsp; Age: {age_cat} &nbsp;|&nbsp; BMI: {bmi}</p>
    </div>
    """, unsafe_allow_html=True)

    # Risk verdict
    if prediction == 1:
        st.markdown(f"""
        <div class="risk-high">
            <b>⚠️ Higher Risk of Heart Disease Detected</b><br>
            {name_display} shows indicators associated with heart disease risk. Please consult a cardiologist.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <b>✅ Lower Risk of Heart Disease</b><br>
            {name_display}'s profile suggests a lower risk. Continue maintaining a healthy lifestyle.
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Three visualisations ─────────────────────────────────────────────────
    v1, v2, v3 = st.columns(3)

    with v1:
        st.markdown('<p class="section-title">Risk Gauge</p>', unsafe_allow_html=True)
        st.pyplot(gauge_chart(prob), use_container_width=True)

    with v2:
        st.markdown('<p class="section-title">Risk Factors Present</p>', unsafe_allow_html=True)
        st.pyplot(risk_factor_bar(input_df), use_container_width=True)

    with v3:
        st.markdown('<p class="section-title">Health Profile Radar</p>', unsafe_allow_html=True)
        st.pyplot(health_radar(physical_health, mental_health, bmi, sleep_time, gh_idx),
                  use_container_width=True)

    # ── Summary metrics ──────────────────────────────────────────────────────
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score",        f"{prob*100:.1f}%")
    m2.metric("BMI",               f"{bmi:.1f}")
    m3.metric("Sleep",             f"{sleep_time} hrs")
    m4.metric("Physical Health",   f"{physical_health}/30 bad days")

  

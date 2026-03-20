import streamlit as st
import pandas as pd
import json
from datetime import datetime
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from groq import Groq

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide"
)

# ── Custom CSS – Full Black Theme ────────────────────────────────────────────
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"],
    .main, .block-container {
        background-color: #0a0a0a !important;
        color: #e8e8e8 !important;
    }
    [data-testid="stSidebar"] { background-color: #111111 !important; }

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] h1,
    [data-testid="stMarkdownContainer"] h2,
    [data-testid="stMarkdownContainer"] h3 {
        color: #f0f0f0 !important;
    }

    label, .stSelectbox label, .stSlider label,
    .stNumberInput label, .stTextInput label,
    .stCheckbox label, p, span {
        color: #cccccc !important;
    }

    .stTextInput input, .stNumberInput input {
        background-color: #1a1a1a !important;
        color: #e8e8e8 !important;
        border: 1px solid #333 !important;
        border-radius: 6px !important;
    }
    .stSelectbox > div > div {
        background-color: #1a1a1a !important;
        color: #e8e8e8 !important;
        border: 1px solid #333 !important;
    }
    [data-baseweb="select"] * { color: #e8e8e8 !important; }
    [data-baseweb="popover"] { background-color: #1a1a1a !important; }
    .stSlider [data-testid="stSlider"] { color: #e63946 !important; }
    .stCheckbox input[type="checkbox"] { accent-color: #e63946; }
    hr { border-color: #2a2a2a !important; }

    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #e63946, #c1121f) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 16px !important;
        padding: 14px !important;
        transition: opacity 0.2s !important;
    }
    .stFormSubmitButton > button:hover { opacity: 0.88 !important; }

    [data-testid="metric-container"] {
        background-color: #141414 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 10px !important;
        padding: 16px !important;
    }
    [data-testid="metric-container"] label { color: #999 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #f0f0f0 !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    .patient-header {
        background: linear-gradient(135deg, #c1121f, #7d0000);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #3a0000;
    }
    .patient-header h2 { margin: 0; font-size: 26px; }
    .patient-header p  { margin: 4px 0 0; opacity: 0.85; font-size: 14px; }

    .risk-high {
        background: #1a0505;
        border-left: 5px solid #e63946;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #ffb3b3;
    }
    .risk-low {
        background: #051a0d;
        border-left: 5px solid #2d6a4f;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 12px;
        color: #90e0b0;
    }

    .section-title {
        font-size: 13px;
        font-weight: 700;
        color: #888 !important;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    [data-testid="stForm"] {
        background-color: #111111 !important;
        border: 1px solid #222 !important;
        border-radius: 12px !important;
        padding: 24px !important;
    }

    /* ── AI Tips Panel ── */
    .ai-tips-container {
        background: linear-gradient(145deg, #0f0f1a, #111122);
        border: 1px solid #2a2a4a;
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 8px;
    }
    .ai-tips-title {
        font-size: 18px;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #e63946);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 18px;
        display: block;
    }
    .tip-card {
        background: #16162a;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: #d4d4e8;
        font-size: 14px;
        line-height: 1.6;
    }
    .tip-card b { color: #a78bfa; }

    /* ── Patient History Table ── */
    .history-container {
        background: #0f0f0f;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 22px 26px;
        margin-top: 8px;
    }
    .history-title {
        font-size: 17px;
        font-weight: 700;
        color: #f0f0f0 !important;
        margin-bottom: 16px;
        display: block;
    }
    .history-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: #161616;
        border: 1px solid #222;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 8px;
        font-size: 13px;
        color: #cccccc;
    }
    .history-row:hover { border-color: #444; }
    .badge-high {
        background: #3a0a0a;
        color: #e63946;
        border: 1px solid #e63946;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        font-weight: 700;
    }
    .badge-low {
        background: #0a3a1a;
        color: #52b788;
        border: 1px solid #52b788;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 11px;
        font-weight: 700;
    }
    .clear-btn button {
        background: #1a1a1a !important;
        color: #e63946 !important;
        border: 1px solid #e63946 !important;
        border-radius: 6px !important;
        font-size: 12px !important;
        padding: 4px 14px !important;
    }

    .disclaimer {
        font-size: 11px;
        color: #555 !important;
        margin-top: 14px;
        font-style: italic;
    }

    /* ── What-If Simulator ── */
    .simulator-container {
        background: linear-gradient(160deg, #0a0a14, #0f0f1e);
        border: 1px solid #1e1e3a;
        border-radius: 16px;
        padding: 28px 32px;
        margin-top: 8px;
    }
    .simulator-title {
        font-size: 20px;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #e63946);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-bottom: 6px;
    }
    .simulator-subtitle {
        font-size: 13px;
        color: #666 !important;
        margin-bottom: 22px;
        display: block;
    }
    .sim-result-box {
        background: #0d0d1a;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        margin-top: 8px;
    }
    .sim-orig-score {
        font-size: 14px;
        color: #888 !important;
        margin-bottom: 4px;
    }
    .sim-new-score {
        font-size: 42px;
        font-weight: 900;
        color: #60a5fa;
        line-height: 1.1;
    }
    .sim-drop-positive {
        font-size: 18px;
        font-weight: 700;
        color: #52b788 !important;
        margin-top: 6px;
    }
    .sim-drop-negative {
        font-size: 18px;
        font-weight: 700;
        color: #e63946 !important;
        margin-top: 6px;
    }
    .sim-insight {
        background: #12121f;
        border-left: 3px solid #60a5fa;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 13px;
        color: #b0b8d0 !important;
        text-align: left;
    }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib global dark style ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d0d0d",
    "axes.facecolor":    "#0d0d0d",
    "savefig.facecolor": "#0d0d0d",
    "text.color":        "#e8e8e8",
    "axes.labelcolor":   "#e8e8e8",
    "xtick.color":       "#aaaaaa",
    "ytick.color":       "#aaaaaa",
    "axes.edgecolor":    "#333333",
    "grid.color":        "#2a2a2a",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Constants ─────────────────────────────────────────────────────────────────
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

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_rf_model.pkl")

# ── Encoding ──────────────────────────────────────────────────────────────────
def encode_inputs(bmi, smoking, alcohol, stroke, physical_health,
                  mental_health, walking_diff, gender, age_cat,
                  ethnicity, diabetic, physical_activity,
                  general_health, sleep_time, kidney_disease):
    age_map       = {cat: i for i, cat in enumerate(AGE_CATEGORIES)}
    ethnicity_map = {e: i for i, e in enumerate(ETHNICITIES)}
    health_map    = {h: i for i, h in enumerate(GENERAL_HEALTH)}
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

# ── AI Health Tips (Groq - Free) ─────────────────────────────────────────────
def get_ai_health_tips(patient_data: dict, risk_score: float) -> str:
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    flags = []
    if patient_data["smoking"]:               flags.append("smoker")
    if patient_data["alcohol"]:               flags.append("heavy alcohol drinker")
    if patient_data["stroke"]:                flags.append("history of stroke")
    if patient_data["diabetic"]:              flags.append("diabetic")
    if patient_data["kidney_disease"]:        flags.append("kidney disease")
    if patient_data["walking_diff"]:          flags.append("difficulty walking")
    if not patient_data["physical_activity"]: flags.append("physically inactive")

    bmi_cat = (
        "underweight" if patient_data["bmi"] < 18.5 else
        "normal weight" if patient_data["bmi"] < 25 else
        "overweight" if patient_data["bmi"] < 30 else "obese"
    )

    prompt = f"""You are a compassionate preventive cardiology health coach.

Patient Profile:
- Age group: {patient_data['age_cat']}, Gender: {patient_data['gender']}
- BMI: {patient_data['bmi']:.1f} ({bmi_cat})
- Sleep: {patient_data['sleep_time']} hrs/night
- General health: {patient_data['general_health']}
- Poor physical health days (last 30): {patient_data['physical_health']}
- Poor mental health days (last 30): {patient_data['mental_health']}
- Risk factors present: {', '.join(flags) if flags else 'none identified'}
- Heart disease risk score: {risk_score*100:.1f}%

Give exactly 5 highly personalised, actionable lifestyle tips to improve this patient's heart health.
Each tip must directly address their specific profile above.
Format each tip strictly as one line:
ICON | **Bold Title** | One clear sentence of advice.

Use these icons in order: 🏃 🥗 😴 🧘 💊
Be warm, specific, and motivating. No generic advice.
Output only the 5 lines — no intro, no outro, no extra text."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
    )
    return response.choices[0].message.content


def render_tips_html(tips_text: str) -> str:
    lines = [l.strip() for l in tips_text.strip().split("\n") if l.strip()]
    cards_html = ""
    for line in lines:
        parts = line.split("|", 2)
        if len(parts) == 3:
            icon   = parts[0].strip()
            title  = parts[1].strip().strip("**").strip("*")
            advice = parts[2].strip()
            cards_html += f'<div class="tip-card"><span style="font-size:18px">{icon}</span> <b>{title}</b><br>{advice}</div>'
        else:
            cards_html += f'<div class="tip-card">{line}</div>'
    return cards_html



# ── What-If Simulator ─────────────────────────────────────────────────────────
def whatif_chart(original_prob, scenarios):
    """Bar chart comparing original risk vs what-if scenarios."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    labels = ["Current"] + [s["label"] for s in scenarios]
    values = [original_prob * 100] + [s["prob"] * 100 for s in scenarios]
    colors = []
    for i, v in enumerate(values):
        if i == 0:
            colors.append("#555577")
        elif v < original_prob * 100:
            colors.append("#52b788")
        else:
            colors.append("#e63946")

    bars = ax.barh(labels, values, color=colors, height=0.5, edgecolor="none")
    ax.set_xlim(0, 105)
    ax.axvline(50, color="#444", linewidth=0.8, linestyle="--")
    ax.text(51, -0.6, "50%", fontsize=8, color="#555")

    for bar, val in zip(bars, values):
        ax.text(val + 1.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9,
                color="#e8e8e8", fontweight="600")

    ax.set_xlabel("Predicted Risk Score (%)", color="#aaaaaa", fontsize=10)
    ax.tick_params(axis="y", labelsize=10, colors="#cccccc")
    ax.tick_params(axis="x", colors="#777777")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.set_title("What-If: How Lifestyle Changes Affect Your Risk",
                 color="#f0f0f0", fontsize=12, fontweight="700", pad=12)
    plt.tight_layout(pad=0.8)
    return fig

# ── Charts ────────────────────────────────────────────────────────────────────
BG = "#0d0d0d"

def gauge_chart(prob):
    fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw=dict(aspect="equal"))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    theta = np.linspace(np.pi, 0, 200)
    for i in range(len(theta) - 1):
        t = i / (len(theta) - 1)
        color = (0.18 + 0.72 * t, 0.60 - 0.47 * t, 0.18)
        ax.plot([np.cos(theta[i]), np.cos(theta[i+1])],
                [np.sin(theta[i]), np.sin(theta[i+1])],
                color=color, linewidth=14, solid_capstyle="butt")
    angle = np.pi - prob * np.pi
    ax.annotate("", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="#f0f0f0",
                                lw=2.5, mutation_scale=18))
    ax.add_patch(plt.Circle((0, 0), 0.08, color="#f0f0f0", zorder=5))
    ax.text(0, -0.22, f"{prob*100:.1f}%", ha="center", va="center",
            fontsize=22, fontweight="bold", color="#f0f0f0")
    ax.text(-1.0, -0.28, "Low",  fontsize=9, color="#52b788", fontweight="600")
    ax.text( 0.72, -0.28, "High", fontsize=9, color="#e63946", fontweight="600")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.4, 1.1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def risk_factor_bar(input_row):
    factors = {
        "Smoking":           input_row["Smoking"].values[0],
        "Alcohol":           input_row["AlcoholDrinking"].values[0],
        "Stroke":            input_row["Stroke"].values[0],
        "Diabetic":          input_row["Diabetic"].values[0],
        "Kidney Disease":    input_row["KidneyDisease"].values[0],
        "Walking Diff.":     input_row["WalkingDifficulty"].values[0],
        "Physical Activity": input_row["PhysicalActivity"].values[0],
    }
    labels = list(factors.keys())
    values = [int(v) for v in factors.values()]
    colors = ["#e63946" if v == 1 else "#2a2a2a" for v in values]
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.barh(labels, values, color=colors, height=0.5, edgecolor="none")
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No", "Yes"], fontsize=10, color="#aaaaaa")
    ax.tick_params(axis="y", labelsize=10, colors="#cccccc")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    red_patch  = mpatches.Patch(color="#e63946", label="Present")
    gray_patch = mpatches.Patch(color="#2a2a2a", label="Absent",
                                edgecolor="#555", linewidth=0.8)
    ax.legend(handles=[red_patch, gray_patch], fontsize=8,
              loc="lower right", framealpha=0, labelcolor="#cccccc")
    plt.tight_layout(pad=0.5)
    return fig


def health_radar(physical_health, mental_health, bmi, sleep_time, general_health_idx):
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#111111")
    categories = ["Physical\nHealth", "Mental\nHealth", "BMI\nRisk",
                  "Sleep\nQuality", "Gen.\nHealth"]
    values = [
        physical_health / 30,
        mental_health / 30,
        min(max((bmi - 18.5) / 21.5, 0), 1),
        1 - min(sleep_time / 9, 1),
        1 - general_health_idx / 4,
    ]
    values += values[:1]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8.5, color="#cccccc")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""], fontsize=7)
    ax.grid(color="#2a2a2a", linewidth=0.8)
    ax.spines["polar"].set_color("#2a2a2a")
    ax.plot(angles, values, color="#e63946", linewidth=2)
    ax.fill(angles, values, color="#e63946", alpha=0.30)
    plt.tight_layout(pad=0.5)
    return fig

# ── App ───────────────────────────────────────────────────────────────────────
st.title("🫀 Heart Disease Risk Predictor")

model = load_model()

with st.form("prediction_form"):
    st.subheader("Patient Details")
    patient_name = st.text_input("Patient Name", placeholder="e.g. Rahul Sharma")

    st.divider()
    st.subheader("Personal Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender    = st.selectbox("Gender", ["Male", "Female"])
        age_cat   = st.selectbox("Age Category", AGE_CATEGORIES, index=7)
        ethnicity = st.selectbox("Ethnicity", ETHNICITIES, index=5)

    with col2:
        bmi            = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
        sleep_time     = st.number_input("Avg Sleep (hrs/night)", min_value=1.0, max_value=24.0, value=7.0, step=0.5)
        general_health = st.selectbox("General Health", GENERAL_HEALTH, index=3)

    with col3:
        physical_health = st.slider("Poor Physical Health Days (last 30)", 0, 30, 0)
        mental_health   = st.slider("Poor Mental Health Days (last 30)", 0, 30, 0)

    st.subheader("Medical History")
    col4, col5, col6 = st.columns(3)

    with col4:
        smoking = st.checkbox("Smoker")
        alcohol = st.checkbox("Heavy Alcohol Drinker")
        stroke  = st.checkbox("Had a Stroke")

    with col5:
        diabetic       = st.checkbox("Diabetic")
        kidney_disease = st.checkbox("Kidney Disease")
        walking_diff   = st.checkbox("Difficulty Walking")

    with col6:
        physical_activity = st.checkbox("Physically Active (last 30 days)", value=True)

    submitted = st.form_submit_button("🔍 Analyse Risk", use_container_width=True)

# ── Results ───────────────────────────────────────────────────────────────────
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

    # Patient header
    st.markdown(f"""
    <div class="patient-header">
        <h2>📋 Report for {name_display}</h2>
        <p>{gender} &nbsp;|&nbsp; Age: {age_cat} &nbsp;|&nbsp; BMI: {bmi}</p>
    </div>
    """, unsafe_allow_html=True)

    # Verdict
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

    # Visualisations
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

    # Summary metrics
    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk Score",      f"{prob*100:.1f}%")
    m2.metric("BMI",             f"{bmi:.1f}")
    m3.metric("Sleep",           f"{sleep_time} hrs")
    m4.metric("Physical Health", f"{physical_health}/30 bad days")

    # ── ✨ AI-Powered Personalised Health Tips ─────────────────────────────────
    st.divider()

    patient_data = dict(
        bmi=bmi, smoking=smoking, alcohol=alcohol, stroke=stroke,
        physical_health=physical_health, mental_health=mental_health,
        walking_diff=walking_diff, gender=gender, age_cat=age_cat,
        diabetic=diabetic, physical_activity=physical_activity,
        general_health=general_health, sleep_time=sleep_time,
        kidney_disease=kidney_disease,
    )

    with st.spinner("✨ Generating personalised AI health tips…"):
        try:
            tips_text  = get_ai_health_tips(patient_data, prob)
            cards_html = render_tips_html(tips_text)
            st.markdown(f"""
            <div class="ai-tips-container">
                <span class="ai-tips-title">✨ AI-Powered Personalised Health Tips for {name_display}</span>
                {cards_html}
                <p class="disclaimer">
                    ⚕️ These AI-generated tips are for informational purposes only and do not constitute
                    medical advice. Always consult a qualified healthcare professional before making
                    changes to your lifestyle or treatment plan.
                </p>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Could not load AI tips: {e}")


    # ── 🧠 What-If Simulator ───────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    <div class="simulator-container">
        <span class="simulator-title">🧠 What-If Simulator</span>
        <span class="simulator-subtitle">Adjust lifestyle factors below and instantly see how your heart disease risk changes using the same ML model.</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Open What-If Simulator", expanded=True):
        st.markdown("**Tweak the sliders to simulate lifestyle improvements:**")

        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            sim_bmi       = st.slider("💪 Simulate BMI",        10.0, 60.0, float(bmi),        0.5, key="sim_bmi")
            sim_sleep     = st.slider("😴 Simulate Sleep (hrs)", 1.0,  12.0, float(sleep_time), 0.5, key="sim_sleep")
        with wc2:
            sim_phys      = st.slider("🏃 Physical Health Bad Days", 0, 30, physical_health, 1, key="sim_phys")
            sim_mental    = st.slider("🧘 Mental Health Bad Days",   0, 30, mental_health,   1, key="sim_mental")
        with wc3:
            sim_smoking   = st.checkbox("🚬 Still Smoking",         value=smoking,          key="sim_smoke")
            sim_active    = st.checkbox("🏋️ Physically Active",     value=physical_activity, key="sim_active")
            sim_alcohol   = st.checkbox("🍺 Heavy Alcohol Drinker", value=alcohol,           key="sim_alc")
            sim_gh        = st.selectbox("🩺 General Health",       GENERAL_HEALTH,
                                         index=GENERAL_HEALTH.index(general_health), key="sim_gh")

        # Build 4 preset scenarios + 1 custom
        scenarios = []

        # Scenario 1: Quit smoking (if smoker)
        if smoking:
            df_s1 = encode_inputs(bmi, False, alcohol, stroke, physical_health,
                                  mental_health, walking_diff, gender, age_cat,
                                  ethnicity, diabetic, physical_activity,
                                  general_health, sleep_time, kidney_disease)
            scenarios.append({"label": "Quit Smoking", "prob": model.predict_proba(df_s1)[0][1]})

        # Scenario 2: Improve sleep to 8hrs
        if sleep_time < 7.5:
            df_s2 = encode_inputs(bmi, smoking, alcohol, stroke, physical_health,
                                  mental_health, walking_diff, gender, age_cat,
                                  ethnicity, diabetic, physical_activity,
                                  general_health, 8.0, kidney_disease)
            scenarios.append({"label": "Sleep 8 hrs", "prob": model.predict_proba(df_s2)[0][1]})

        # Scenario 3: Reduce BMI by 3
        if bmi > 22:
            df_s3 = encode_inputs(max(18.5, bmi - 3), smoking, alcohol, stroke,
                                  physical_health, mental_health, walking_diff,
                                  gender, age_cat, ethnicity, diabetic,
                                  physical_activity, general_health, sleep_time, kidney_disease)
            scenarios.append({"label": f"BMI -{3}", "prob": model.predict_proba(df_s3)[0][1]})

        # Scenario 4: Become physically active
        if not physical_activity:
            df_s4 = encode_inputs(bmi, smoking, alcohol, stroke, physical_health,
                                  mental_health, walking_diff, gender, age_cat,
                                  ethnicity, diabetic, True,
                                  general_health, sleep_time, kidney_disease)
            scenarios.append({"label": "Get Active", "prob": model.predict_proba(df_s4)[0][1]})

        # Scenario 5: Best case — all improvements
        df_best = encode_inputs(
            max(18.5, bmi - 3), False, False, stroke,
            max(0, physical_health - 5), max(0, mental_health - 5),
            walking_diff, gender, age_cat, ethnicity, diabetic,
            True, "Very good", min(9.0, sleep_time + 1.5), kidney_disease
        )
        scenarios.append({"label": "Best Case 🌟", "prob": model.predict_proba(df_best)[0][1]})

        # Scenario 6: Custom (user sliders)
        df_custom = encode_inputs(
            sim_bmi, sim_smoking, sim_alcohol, stroke, sim_phys,
            sim_mental, walking_diff, gender, age_cat, ethnicity,
            diabetic, sim_active, sim_gh, sim_sleep, kidney_disease
        )
        custom_prob = model.predict_proba(df_custom)[0][1]
        scenarios.append({"label": "🎛️ My Custom", "prob": custom_prob})

        # ── Results ──
        st.divider()
        rc1, rc2 = st.columns([2, 1])

        with rc1:
            if scenarios:
                st.pyplot(whatif_chart(prob, scenarios), use_container_width=True)

        with rc2:
            # Show custom scenario result prominently
            delta     = prob - custom_prob
            delta_pct = round(delta * 100, 1)
            drop_class = "sim-drop-positive" if delta > 0 else "sim-drop-negative"
            arrow      = "▼" if delta > 0 else "▲"
            change_txt = f"{arrow} {abs(delta_pct)}% {'reduction' if delta > 0 else 'increase'} in risk"

            # Insight message
            insights = []
            if sim_bmi < bmi - 1:       insights.append(f"Lowering BMI from {bmi} → {sim_bmi} helps reduce cardiac load.")
            if sim_sleep > sleep_time:  insights.append(f"Better sleep ({sim_sleep}h) improves heart recovery cycles.")
            if not sim_smoking and smoking: insights.append("Quitting smoking is the single highest-impact change.")
            if sim_active and not physical_activity: insights.append("Becoming active significantly lowers risk.")
            insight_html = "<br>".join(f"• {i}" for i in insights) if insights else "• Adjust the sliders above to see personalised insights."

            st.markdown(f"""
            <div class="sim-result-box">
                <p class="sim-orig-score">Original Risk: {prob*100:.1f}%</p>
                <p class="sim-new-score">{custom_prob*100:.1f}%</p>
                <p class="{drop_class}">{change_txt}</p>
                <div class="sim-insight">{insight_html}</div>
            </div>
            """, unsafe_allow_html=True)

            # Best case callout
            best_prob = model.predict_proba(df_best)[0][1]
            best_drop = round((prob - best_prob) * 100, 1)
            if best_drop > 0:
                st.success(f"🌟 With all improvements, risk could drop by **{best_drop}%** (to {best_prob*100:.1f}%)")

    # ── 📊 Save to Patient History ─────────────────────────────────────────────
    if "patient_history" not in st.session_state:
        st.session_state.patient_history = []

    record = {
        "name":       name_display,
        "time":       datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "gender":     gender,
        "age":        age_cat,
        "bmi":        round(bmi, 1),
        "sleep":      sleep_time,
        "risk_score": round(prob * 100, 1),
        "prediction": int(prediction),
        "smoking":    smoking,
        "diabetic":   diabetic,
        "physical_health": physical_health,
        "mental_health":   mental_health,
        "general_health":  general_health,
    }
    # Avoid duplicate on re-run
    if not st.session_state.patient_history or st.session_state.patient_history[-1] != record:
        st.session_state.patient_history.append(record)

# ── 📊 Patient History Section ────────────────────────────────────────────────
if "patient_history" in st.session_state and len(st.session_state.patient_history) > 0:
    history = st.session_state.patient_history

    st.divider()
    st.subheader("📊 Patient History")

    # ── Clear button ──
    if st.button("🗑️ Clear History"):
        st.session_state.patient_history = []
        st.rerun()

    # ── History rows ──
    rows_html = ""
    for i, r in enumerate(reversed(history)):
        badge = f'<span class="badge-high">⚠ High Risk</span>' if r["prediction"] == 1 else f'<span class="badge-low">✓ Low Risk</span>'
        rows_html += f"""
        <div class="history-row">
            <span><b style="color:#f0f0f0">{r['name']}</b> &nbsp;·&nbsp; {r['gender']} &nbsp;·&nbsp; {r['age']}</span>
            <span>BMI {r['bmi']} &nbsp;|&nbsp; Sleep {r['sleep']}h &nbsp;|&nbsp; {r['time']}</span>
            <span>Risk: <b style="color:#e8e8e8">{r['risk_score']}%</b> &nbsp;{badge}</span>
        </div>"""

    st.markdown(f"""
    <div class="history-container">
        <span class="history-title">👥 All Analysed Patients — {len(history)} record(s)</span>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

    # ── 📈 Risk Trend Chart ────────────────────────────────────────────────────
    if len(history) >= 2:
        st.divider()
        st.subheader("📈 Risk Trend Chart")
        st.caption("Compares risk scores across all analysed patients this session.")

        names       = [f"{r['name']} ({r['age']})" for r in history]
        risk_scores = [r["risk_score"] for r in history]
        bmis        = [r["bmi"] for r in history]
        sleeps      = [r["sleep"] for r in history]

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        fig.patch.set_facecolor("#0d0d0d")

        # ── Chart 1: Risk Score per Patient ──
        ax1 = axes[0]
        ax1.set_facecolor("#0d0d0d")
        bar_colors = ["#e63946" if r["prediction"] == 1 else "#52b788" for r in history]
        bars = ax1.bar(range(len(names)), risk_scores, color=bar_colors,
                       edgecolor="none", width=0.55)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels([r["name"] for r in history], rotation=25,
                            ha="right", fontsize=9, color="#cccccc")
        ax1.set_ylabel("Risk Score (%)", color="#aaaaaa", fontsize=10)
        ax1.set_ylim(0, 105)
        ax1.axhline(50, color="#555", linewidth=0.8, linestyle="--")
        ax1.text(len(names) - 0.5, 52, "50% threshold", fontsize=8,
                 color="#666", ha="right")
        for bar, val in zip(bars, risk_scores):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                     f"{val}%", ha="center", va="bottom", fontsize=8,
                     color="#e8e8e8", fontweight="600")
        ax1.set_title("Risk Score per Patient", color="#f0f0f0",
                      fontsize=12, fontweight="700", pad=12)
        ax1.spines["bottom"].set_color("#333")
        ax1.spines["left"].set_color("#333")
        ax1.tick_params(colors="#aaaaaa")

        # ── Chart 2: Risk Score Trend Line ──
        ax2 = axes[1]
        ax2.set_facecolor("#0d0d0d")
        x = range(len(history))
        ax2.plot(x, risk_scores, color="#e63946", linewidth=2.5,
                 marker="o", markersize=7, markerfacecolor="#ff6b6b",
                 markeredgecolor="#e63946", zorder=3)
        ax2.fill_between(x, risk_scores, alpha=0.15, color="#e63946")
        ax2.set_xticks(x)
        ax2.set_xticklabels([r["name"] for r in history], rotation=25,
                            ha="right", fontsize=9, color="#cccccc")
        ax2.set_ylabel("Risk Score (%)", color="#aaaaaa", fontsize=10)
        ax2.set_ylim(0, 105)
        ax2.axhline(50, color="#555", linewidth=0.8, linestyle="--")
        for xi, val in zip(x, risk_scores):
            ax2.annotate(f"{val}%", (xi, val), textcoords="offset points",
                         xytext=(0, 10), ha="center", fontsize=8,
                         color="#e8e8e8")
        ax2.set_title("Risk Score Trend", color="#f0f0f0",
                      fontsize=12, fontweight="700", pad=12)
        ax2.spines["bottom"].set_color("#333")
        ax2.spines["left"].set_color("#333")
        ax2.tick_params(colors="#aaaaaa")
        ax2.grid(axis="y", color="#1e1e1e", linewidth=0.8)

        # ── Chart 3: BMI vs Sleep Scatter ──
        ax3 = axes[2]
        ax3.set_facecolor("#0d0d0d")
        sc_colors = ["#e63946" if r["prediction"] == 1 else "#52b788" for r in history]
        scatter = ax3.scatter(bmis, sleeps, c=sc_colors, s=120,
                              edgecolors="#444", linewidths=0.8, zorder=3)
        for i, r in enumerate(history):
            ax3.annotate(r["name"], (bmis[i], sleeps[i]),
                         textcoords="offset points", xytext=(6, 4),
                         fontsize=8, color="#aaaaaa")
        ax3.set_xlabel("BMI", color="#aaaaaa", fontsize=10)
        ax3.set_ylabel("Sleep (hrs)", color="#aaaaaa", fontsize=10)
        ax3.set_title("BMI vs Sleep Quality", color="#f0f0f0",
                      fontsize=12, fontweight="700", pad=12)
        ax3.spines["bottom"].set_color("#333")
        ax3.spines["left"].set_color("#333")
        ax3.tick_params(colors="#aaaaaa")
        ax3.grid(color="#1e1e1e", linewidth=0.8)

        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e63946",
                   markersize=9, label="High Risk"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#52b788",
                   markersize=9, label="Low Risk"),
        ]
        ax3.legend(handles=legend_elements, fontsize=8, framealpha=0,
                   labelcolor="#cccccc", loc="upper right")

        plt.tight_layout(pad=2.0)
        st.pyplot(fig, use_container_width=True)

        # ── Summary Stats ──
        st.divider()
        s1, s2, s3, s4 = st.columns(4)
        high_risk = sum(1 for r in history if r["prediction"] == 1)
        avg_risk  = round(sum(r["risk_score"] for r in history) / len(history), 1)
        avg_bmi   = round(sum(r["bmi"] for r in history) / len(history), 1)
        avg_sleep = round(sum(r["sleep"] for r in history) / len(history), 1)
        s1.metric("Total Patients",  len(history))
        s2.metric("High Risk Count", f"{high_risk} / {len(history)}")
        s3.metric("Avg Risk Score",  f"{avg_risk}%")
        s4.metric("Avg BMI",         avg_bmi)
    else:
        st.info("📈 Analyse at least 2 patients to see the Risk Trend Chart.")

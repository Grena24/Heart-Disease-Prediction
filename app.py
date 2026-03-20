import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import anthropic

# ✅ Add this AFTER all imports
os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]

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
    .disclaimer {
        font-size: 11px;
        color: #555 !important;
        margin-top: 14px;
        font-style: italic;
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

# ── AI Health Tips ────────────────────────────────────────────────────────────
def get_ai_health_tips(patient_data: dict, risk_score: float) -> str:
    client = anthropic.Anthropic()

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

    full_response = ""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            full_response += text
    return full_response


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

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Failure Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e);
        color: #ffffff;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-right: 1px solid #e74c3c33;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e74c3c, #ff6b6b, #ffd93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #adb5bd;
        font-size: 1.1rem;
    }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(231,76,60,0.3);
        border-radius: 16px;
        padding: 1.2rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #e74c3c;
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        color: #adb5bd;
        font-size: 0.85rem;
        margin: 0;
    }

    /* Result boxes */
    .result-danger {
        background: linear-gradient(135deg, rgba(231,76,60,0.2), rgba(192,57,43,0.1));
        border: 2px solid #e74c3c;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-safe {
        background: linear-gradient(135deg, rgba(46,204,113,0.2), rgba(39,174,96,0.1));
        border: 2px solid #2ecc71;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
    }
    .result-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .result-subtitle {
        font-size: 1rem;
        color: #adb5bd;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e74c3c;
        border-bottom: 2px solid #e74c3c33;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Input labels */
    label {
        color: #e0e0e0 !important;
        font-weight: 500 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #e74c3c, #c0392b);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(231,76,60,0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #c0392b, #e74c3c);
        box-shadow: 0 6px 25px rgba(231,76,60,0.6);
        transform: translateY(-2px);
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255,193,7,0.1);
        border: 1px solid rgba(255,193,7,0.4);
        border-radius: 12px;
        padding: 1rem;
        color: #ffc107;
        font-size: 0.85rem;
        text-align: center;
        margin-top: 1rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Selectbox & inputs */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(231,76,60,0.3) !important;
        border-radius: 10px !important;
        color: white !important;
    }
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(231,76,60,0.3) !important;
        border-radius: 10px !important;
        color: white !important;
    }

    /* Dataframe */
    .dataframe {
        background: rgba(255,255,255,0.05) !important;
    }

    /* Probability bar label */
    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        color: #adb5bd;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LOAD & TRAIN MODEL
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv("cleaned_heart.csv")
    df = df.drop(columns=['AgeGroup', 'BP_Category'])

    cat_features = ['Gender', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    le = LabelEncoder()
    encoders = {}

    for col in cat_features:
        df[col] = le.fit_transform(df[col])
        encoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return model, encoders

model, encoders = load_model()


# ─────────────────────────────────────────────────────────────
# SIDEBAR — ABOUT
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ❤️ About This App")
    st.markdown("""
    This app uses a **Random Forest** machine learning model
    to predict the risk of heart failure based on clinical data.
    """)

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>87.5%</h3>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h3>0.93</h3>
            <p>ROC-AUC</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏥 Dataset Info")
    st.markdown("""
    - 📋 **918** patient records
    - 🔬 **11** clinical features
    - 🎯 Binary classification
    - ⚖️ Balanced classes
    """)

    st.markdown("---")
    st.markdown("### 📌 Chest Pain Types")
    st.markdown("""
    - **ASY** → Asymptomatic
    - **ATA** → Atypical Angina
    - **NAP** → Non-Anginal Pain
    - **TA**  → Typical Angina
    """)

    st.markdown("---")
    st.markdown("""
    <div class='disclaimer'>
        ⚕️ For educational use only.<br>
        Not a substitute for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='main-header'>
    <h1>❤️ Heart Failure Prediction</h1>
    <p>Enter patient clinical data below to assess heart disease risk using AI</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# MAIN — INPUT FORM
# ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🧑‍⚕️ Patient Clinical Information</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**👤 Personal Info**")
    age    = st.slider("Age (years)", 18, 100, 50)
    gender = st.selectbox("Gender", ["Male", "Female"])
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])

with col2:
    st.markdown("**🫀 Heart Metrics**")
    chest      = st.selectbox("Chest Pain Type", ["ASY - Asymptomatic",
                                                    "ATA - Atypical Angina",
                                                    "NAP - Non-Anginal Pain",
                                                    "TA  - Typical Angina"])
    max_hr     = st.slider("Max Heart Rate", 50, 250, 140)
    ex_angina  = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak    = st.number_input("Oldpeak (ST Depression)", 0.0, 10.0, 1.0, step=0.1)

with col3:
    st.markdown("**🩺 Clinical Measurements**")
    resting_bp  = st.number_input("Resting Blood Pressure (mmHg)", 50, 250, 120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", 50, 600, 200)
    resting_ecg = st.selectbox("Resting ECG Result", ["NORMAL", "ST", "LVH"])
    st_slope    = st.selectbox("ST Slope", ["UP", "FLAT", "DOWN"])

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# ENCODE INPUTS
# ─────────────────────────────────────────────────────────────
gender_enc  = encoders['Gender']['M'] if gender == "Male" else encoders['Gender']['F']
chest_enc   = encoders['ChestPainType'][chest.split(" ")[0].strip()]
ecg_enc     = encoders['RestingECG'][resting_ecg]
angina_enc  = encoders['ExerciseAngina']['Y'] if ex_angina == "Yes" else encoders['ExerciseAngina']['N']
slope_enc   = encoders['ST_Slope'][st_slope]
fasting_enc = 1 if fasting_bs == "Yes" else 0

input_data = pd.DataFrame([{
    'Age'                 : age,
    'Gender'              : gender_enc,
    'ChestPainType'       : chest_enc,
    'RestingBloodPressure': float(resting_bp),
    'Cholesterol'         : float(cholesterol),
    'FastingBloodSugar'   : fasting_enc,
    'RestingECG'          : ecg_enc,
    'MaxHR'               : max_hr,
    'ExerciseAngina'      : angina_enc,
    'Oldpeak'             : float(oldpeak),
    'ST_Slope'            : slope_enc
}])

# ─────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────
predict_btn = st.button("🔍  Predict Heart Disease Risk", use_container_width=True)

if predict_btn:

    pred  = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.markdown("<div class='section-header'>📋 Prediction Result</div>", unsafe_allow_html=True)

    # Result display
    res_col1, res_col2 = st.columns([1.5, 1])

    with res_col1:
        if pred == 1:
            st.markdown(f"""
            <div class='result-danger'>
                <div class='result-title'>⚠️ HIGH RISK</div>
                <div style='font-size:1.3rem; color:#e74c3c; font-weight:700;'>Heart Disease Detected</div>
                <div class='result-subtitle'>The model predicts a high probability of heart disease.<br>
                Please consult a cardiologist immediately.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-safe'>
                <div class='result-title'>✅ LOW RISK</div>
                <div style='font-size:1.3rem; color:#2ecc71; font-weight:700;'>No Heart Disease Detected</div>
                <div class='result-subtitle'>The model predicts a low probability of heart disease.<br>
                Maintain a healthy lifestyle and regular checkups.</div>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        # Probability metrics
        st.markdown("#### 📊 Probability Breakdown")

        no_disease_pct = proba[0] * 100
        disease_pct    = proba[1] * 100

        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='color:#2ecc71;'>{no_disease_pct:.1f}%</h3>
            <p>✅ No Disease Probability</p>
        </div>
        <div class='metric-card'>
            <h3 style='color:#e74c3c;'>{disease_pct:.1f}%</h3>
            <p>⚠️ Heart Disease Probability</p>
        </div>
        """, unsafe_allow_html=True)

    # Risk bar
    st.markdown("#### 🎯 Risk Gauge")
    st.markdown(f"""
    <div class='prob-label'>
        <span>✅ No Disease ({proba[0]*100:.1f}%)</span>
        <span>⚠️ Disease ({proba[1]*100:.1f}%)</span>
    </div>
    """, unsafe_allow_html=True)
    st.progress(float(proba[1]))

    st.markdown("---")

    # Patient summary table
    st.markdown("<div class='section-header'>📝 Patient Summary</div>", unsafe_allow_html=True)

    summary_df = pd.DataFrame({
        'Feature'  : ['Age', 'Gender', 'Chest Pain Type', 'Resting BP (mmHg)',
                      'Cholesterol (mg/dl)', 'Fasting Blood Sugar',
                      'Resting ECG', 'Max Heart Rate',
                      'Exercise Angina', 'Oldpeak', 'ST Slope'],
        'Value'    : [age, gender, chest.split("-")[1].strip(), resting_bp,
                      cholesterol, fasting_bs, resting_ecg,
                      max_hr, ex_angina, oldpeak, st_slope],
        'Status'   : ['🟡 Risk Factor' if age > 55 else '🟢 Normal',
                      '🟡 Higher Risk' if gender == 'Male' else '🟢 Normal',
                      '🔴 High Risk' if 'ASY' in chest else '🟢 Normal',
                      '🔴 High' if resting_bp > 140 else '🟢 Normal',
                      '🔴 High' if cholesterol > 240 else '🟢 Normal',
                      '🔴 Yes' if fasting_bs == 'Yes' else '🟢 No',
                      '🟢 Normal' if resting_ecg == 'NORMAL' else '🔴 Abnormal',
                      '🔴 Low' if max_hr < 100 else '🟢 Normal',
                      '🔴 Yes' if ex_angina == 'Yes' else '🟢 No',
                      '🔴 High' if oldpeak > 2 else '🟢 Normal',
                      '🔴 Flat/Down' if st_slope in ['FLAT','DOWN'] else '🟢 Up']
    })

    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Disclaimer
    st.markdown("""
    <div class='disclaimer'>
        ⚕️ <strong>Medical Disclaimer:</strong> This application is for educational and portfolio purposes only.
        It is NOT a substitute for professional medical advice, diagnosis, or treatment.
        Always consult a qualified and certified doctor for any medical decisions.
    </div>
    """, unsafe_allow_html=True)

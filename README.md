# 🫀 Heart Disease Risk Prediction

A machine learning web application that predicts the risk of heart disease using clinical patient data, powered by **Random Forest**, **SHAP Explainable AI**, and **Groq LLaMA 3 AI** for personalized health recommendations.

---

## 🚀 Live Demo

🔗 **[Heart Disease Risk Prediction App](https://heart-disease-prediction-ii5zazrzetbq3b2p3jvxmu.streamlit.app/)**

---

## 📌 Project Overview

Heart disease is one of the leading causes of death worldwide. This project aims to assist doctors and patients in detecting cardiovascular risk early using a trained machine learning model. The system takes 11 clinical parameters as input, predicts the risk level, explains the prediction using SHAP, and generates personalized AI health recommendations — all through a clean and accessible web interface.

---

## 🎯 Features

- ✅ **Heart Disease Prediction** — Random Forest Classifier with 87.5% Accuracy and 0.927 ROC-AUC
- 🔬 **SHAP Explainability** — Visual explanation of which features impacted each prediction
- 🤖 **AI Health Recommendations** — Personalized diet, exercise, and lifestyle advice powered by Groq LLaMA 3
- 📄 **PDF Report Generation** — Downloadable patient report with clinical summary and risk-level advice
- 📊 **Patient Summary Table** — Color-coded normal vs abnormal clinical values
- ⚠️ **3-Level Risk Advice** — Low, Moderate, and High risk guidance in the PDF report

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 87.5% |
| ROC-AUC Score | 0.927 |
| Dataset Size | 918 patient records |
| Features Used | 11 clinical features |
| Train/Test Split | 80% / 20% (Stratified) |

---

## 🗂️ Dataset

| Dataset | Link |
|---------|------|
| Original Dataset (Kaggle) | [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) |
| Cleaned Dataset (GitHub) | [cleaned_heart.csv](https://github.com/Grena24/Heart-Disease-Prediction/blob/main/cleaned_heart.csv) |

---

## 🧠 How It Works

1. Doctor or patient enters **11 clinical parameters** into the web app
2. Inputs are **encoded and passed** to the trained Random Forest model
3. Model outputs **Heart Disease (High Risk) or No Heart Disease (Low Risk)**
4. **SHAP** explains which features contributed most to the prediction
5. **Groq LLaMA 3** generates personalized health recommendations
6. A **PDF report** is generated and available for download

---

## 🏥 Input Features

| Feature | Description |
|---------|-------------|
| Age | Patient age in years |
| Gender | Male / Female |
| Chest Pain Type | ASY / ATA / NAP / TA |
| Resting Blood Pressure | mmHg |
| Cholesterol | mg/dl |
| Fasting Blood Sugar | > 120 mg/dl (Yes/No) |
| Resting ECG | NORMAL / ST / LVH |
| Max Heart Rate | bpm |
| Exercise Induced Angina | Yes / No |
| Oldpeak (ST Depression) | Numeric value |
| ST Slope | UP / FLAT / DOWN |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| Python 3.10+ | Core programming language |
| Streamlit | Web application framework |
| Scikit-learn | Random Forest model & evaluation |
| SHAP | Explainable AI |
| Groq (LLaMA 3) | AI health recommendations |
| Matplotlib | SHAP chart visualization |
| ReportLab | PDF report generation |
| Pandas & NumPy | Data processing |
| Power BI | Data visualization & EDA |
| GitHub | Version control & deployment |

---

## 📁 Project Structure

```
Heart-Disease-Prediction/
│
├── app.py                        # Main Streamlit application
├── cleaned_heart.csv             # Cleaned dataset
├── heart_failure_model.pkl       # Saved ML model (optional)
├── ML_model_for_heart_Disease.ipynb  # Model training notebook
├── Heart_failure_project.ipynb   # EDA notebook
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## ⚙️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Grena24/Heart-Disease-Prediction.git
cd Heart-Disease-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your Groq API Key
Go to Streamlit secrets and add:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
shap
matplotlib
groq
reportlab
```

---

## 📚 References

1. Fedesoriano (2021). Heart Failure Prediction Dataset. Kaggle. https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
2. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017. https://arxiv.org/abs/1705.07874
3. Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324
4. Scikit-learn Developers (2024). Scikit-learn Documentation. https://scikit-learn.org
5. Streamlit Inc. (2024). Streamlit Documentation. https://docs.streamlit.io
6. Groq Inc. (2024). Groq API Documentation. https://console.groq.com/docs
7. Microsoft Power BI (2024). Power BI Documentation. https://learn.microsoft.com/en-us/power-bi
8. ReportLab Inc. (2024). ReportLab PDF Library. https://www.reportlab.com

---

## 👤 Author

**Grena24**
- GitHub: [@Grena24](https://github.com/Grena24)

---

## 📄 License

This project is open source and available for educational purposes.

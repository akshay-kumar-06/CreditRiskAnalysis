# 🏦 Credit Risk Prediction System

> An end-to-end Explainable AI (XAI) dashboard for predicting loan default probabilities and interpreting machine learning decisions.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0.0-111111?style=flat)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43.0-blueviolet?style=flat)](https://shap.readthedocs.io/)

---

## 📋 Overview

In the financial industry, accurately assessing credit risk—the probability that a borrower will default on a loan—is a multi-billion dollar challenge. Traditional institutions have historically relied on rigid credit scorecards that fail to capture non-linear relationships or complex feature interactions. 

This project implements an end-to-end **Credit Risk Prediction System** that bridges the gap between high-performance machine learning and interpretability. Using **XGBoost** as the core predictive engine, the system achieves state-of-the-art predictive accuracy while employing **SHAP (SHapley Additive exPlanations)** to provide mathematically sound, transparent insights into model predictions. This ensures that credit decisions are not only accurate but also regulatory-compliant, explainable, and auditable.

---

## 💻 Demo

*Below is a visual preview of the interactive Streamlit dashboard. Users can adjust borrower profiles in the sidebar to observe real-time risk classification changes and view corresponding SHAP explanation plots.*

![Streamlit Dashboard Demo](https://raw.githubusercontent.com/your-username/credit-risk-prediction/main/assets/demo_screenshot.png) *(Placeholder: Update this path with your hosted image/GIF once deployed)*

---

## ✨ Features

- **Robust End-to-End Pipeline**: Automated raw data cleaning, imputation, transformation, and training workflow.
- **Domain-Specific Feature Engineering**: Extracts high-value features such as:
  - `income_to_installment_ratio`: Measures monthly debt service capacity.
  - `fico_category`: Segments borrowers into standard risk brackets (Poor, Fair, Good, Excellent).
  - `dti_category`: Groups debt-to-income ratios based on lending standards.
  - `delinq_risk_score`: Aggregates historical delinquencies and inquiries into a risk coefficient.
- **Benchmark Model Evaluation**: Trains and benchmarks three distinct architectures:
  - *Baseline*: Logistic Regression (L2 Regularized)
  - *Comparison*: Random Forest Classifier
  - *Production*: Cost-sensitive XGBoost with class-imbalance weighting (`scale_pos_weight`).
- **Explainable AI (XAI)**:
  - *Global Interpretability*: Feature Importance and Summary plots indicating general model drivers.
  - *Local Interpretability*: SHAP Force and Waterfall plots mapping how individual borrower attributes drive default probability up or down.
- **Interactive Streamlit Web Dashboard**: Empowers loan officers to simulate profiles, predict creditworthiness, and export results with explanations.

---

## 🛠️ Tech Stack

| Category | Tool / Library |
| :--- | :--- |
| **Language** | Python (3.10+) |
| **Model Architectures** | XGBoost, Scikit-learn, Imbalanced-learn |
| **Explainable AI (XAI)** | SHAP (SHapley Additive exPlanations) |
| **Interactive Dashboard** | Streamlit |
| **Data Engineering** | Pandas, NumPy |
| **Visualization** | Plotly, Seaborn, Matplotlib |
| **Serialization** | Joblib |

---

## 📐 Project Architecture & Workflow

```
Raw Loan Data (CSV)
       │
       ▼
Preprocessing & Quality Controls (Imputation, Encoding, Rescaling)
       │
       ▼
Feature Engineering (Financial Ratios, Delinquency Risk Scores, FICO Categories)
       │
       ▼
Model Benchmarking & Selection (Logistic Regression vs. Random Forest vs. XGBoost)
       │
       ▼
Model Explainability (SHAP Global & Local Interpretability)
       │
       ▼
Streamlit Interactive UI Dashboard (Lending Decisions & Real-time Explanations)
```

---

## 🚀 Installation

Follow these steps to set up the project locally.

### Prerequisites
- Python 3.10 or higher
- Git

### Setup
1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/credit-risk-prediction.git
   cd credit-risk-prediction
   ```

2. **Create and Activate a Virtual Environment**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux/macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 💡 Usage

### 1. Preprocess Raw Data
Runs the preprocessing script to clean the data, engineer domain features, and save processed splits to `data/processed/`.
```bash
python src/preprocessing.py
```

### 2. Model Training & Evaluation
Trains the benchmark models (Logistic Regression, Random Forest, XGBoost), performs evaluation on validation/test sets, generates ROC curves, and serializes artifacts to `models/`.
```bash
python src/model.py
```

### 3. Launch Streamlit Application
Start the interactive dashboard local server:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the dashboard.

---

## 📊 Model Details & Performance

### Algorithm Architecture
The primary production model is **XGBoost (Extreme Gradient Boosting)**, a decision-tree ensemble algorithm using gradient descent. To address the severe class imbalance (non-defaulters vs. defaulters), we configure the `scale_pos_weight` parameter dynamically based on the inverse class ratio.

### Key Input Features
The model consumes 23 input features, including:
- **Demographics & Profile**: `purpose` (loan intent, e.g., credit card, debt consolidation), `log_annual_inc` (logged annual income).
- **Financial Status**: `dti` (debt-to-income ratio), `int_rate` (interest rate), `installment` (monthly payment).
- **Credit History**: `fico` (FICO score), `days_with_cr_line` (credit history length), `revol_bal` & `revol_util` (revolving balance/utilization rate).
- **Risk Indicators**: `inq_last_6mths` (recent inquiries), `delinq_2yrs` (past delinquencies), `pub_rec` (public derogatory records).
- **Engineered Features**: `income_to_installment_ratio`, `delinq_risk_score`, `credit_age_years`.

### Evaluation Metrics (Test Set Benchmarks)

| Metric | Logistic Regression (Baseline) | Random Forest | XGBoost (Production) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | `[X]%` | `[X]%` | `[X]%` |
| **ROC-AUC** | `[X]%` | `[X]%` | `[X]%` |
| **Precision (Default)** | `[X]%` | `[X]%` | `[X]%` |
| **Recall (Default)** | `[X]%` | `[X]%` | `[X]%` |
| **F1-Score (Default)** | `[X]%` | `[X]%` | `[X]%` |

*(Note: Placeholders represent values to be filled following validation on the target dataset)*

---

## 🔍 Explainability & Interpretability (SHAP)

To prevent the model from operating as a "black box," the system leverages **SHAP (SHapley Additive exPlanations)** based on game theory to explain output decisions.

- **Global Interpretability**: 
  - Uses SHAP **Summary Plots** to reveal which features have the highest predictive power across the entire dataset. For instance, FICO scores, inquiries, and DTI ratio generally emerge as dominant features.
- **Local Interpretability**:
  - Incorporates **Force Plots** and **Waterfall Plots** in the Streamlit app. When a user queries a borrower profile, SHAP quantifies how each specific attribute (e.g., a low FICO score of 620 driving the risk up, or a stable income driving the risk down) shifts the final default probability from the model's base value.

---

## 📂 Folder Structure

```
credit_risk/
├── data/
│   ├── raw/                       # Raw loan dataset CSVs
│   └── processed/                 # Train/val/test splits & preprocessed artifacts
├── models/                        # Serialized model (.pkl) files & ROC curves
├── notebooks/                     # Jupyter notebooks for EDA and exploration
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py           # Data quality controls & feature engineering
│   ├── model.py                   # Model training, benchmarking & grid search
│   └── explain.py                 # SHAP interpretation utility functions
├── app.py                         # Streamlit dashboard interface
├── requirements.txt               # Project dependency versions
└── README.md                      # Project documentation and guide
```

---

## 🔮 Future Improvements

- [ ] **Advanced Imbalance Handling**: Compare SMOTE and ADASYN against the current cost-sensitive learning approach.
- [ ] **Hyperparameter Tuning**: Integrate Optuna for automated Bayesian optimization.
- [ ] **Model Monitoring**: Set up Evidently AI or custom pipelines to track data and concept drift.
- [ ] **Dockerization**: Containerize the Streamlit app to facilitate seamless cloud deployment (e.g., AWS ECS, GCP Cloud Run).
- [ ] **CI/CD Pipeline**: Implement GitHub Actions to run automated testing on data validation and code formatting.

---

## 👤 Author

- **Akshaykumar** - *AI/ML Engineer*
  - GitHub: [@your-username](https://github.com/your-username)
  - LinkedIn: [Your Name](https://linkedin.com/in/your-username)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

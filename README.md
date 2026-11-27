# 🏦 Loan Default Prediction & Credit Risk Analysis

### Machine Learning | Financial Risk Assessment | Predictive Analytics

A Machine Learning project focused on analyzing credit risk and predicting the likelihood of loan default using applicant financial and demographic information.

The project explores how historical loan data can be used to develop predictive models that support risk assessment and data-driven lending decisions.

---

## 🎯 Project Overview

Loan defaults create financial risks for banks and lending institutions. Identifying applicants who may have difficulty repaying loans can help institutions make more informed lending decisions.

This project explores a binary classification approach to loan default prediction using a credit risk dataset.

**Main Objective:** Analyze applicant information and estimate the probability of loan default using Machine Learning.

---

## 🔍 Key Project Areas

| Area | Description |
|---|---|
| Data Understanding | Explore dataset structure and variables |
| Data Preprocessing | Prepare data for machine learning |
| Exploratory Data Analysis | Investigate patterns and relationships |
| Feature Engineering | Prepare relevant predictive features |
| Model Development | Develop a classification model |
| Model Evaluation | Evaluate predictive performance |
| Prediction Application | Provide an interactive prediction interface |

*The exact algorithms and evaluation results will be documented after verification of the notebook.*

---

## 📂 Project Structure

```text
loan-default-prediction/
│
├── data/
│   └── credit_risk_dataset.csv
│
├── notebooks/
│   └── loan_default_prediction.ipynb
│
├── models/
│   └── model.pkl
│
├── app/
│   └── app.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 📊 Dataset Information

**Dataset:** Credit Risk Dataset

The dataset contains applicant-related information used for credit risk analysis.

Features used by the current prediction application include:

- Applicant age
- Annual income
- Home ownership
- Employment length
- Loan intent
- Loan grade
- Loan amount
- Interest rate
- Previous default history
- Credit history length

**Target Variable:** `loan_status`

The prediction task is to estimate the likelihood of a loan default.

---

## 🧰 Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (verify notebook implementation) |
| Visualization | Matplotlib, Seaborn (verify notebook implementation) |
| Application | Streamlit |
| Model Loading | Joblib |
| Development | Jupyter Notebook, Git, GitHub |

---

## 🖥️ Interactive Prediction Application

The project includes a Streamlit application designed to collect applicant information and display a predicted default probability.

The interface includes:

- Applicant information inputs
- Financial and employment details
- Loan characteristics
- Credit history information
- Default probability output
- Risk classification display

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/girish2004bagdi/loan-default-prediction.git

cd loan-default-prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Jupyter Notebook

```bash
jupyter notebook
```

Open the notebook inside the `notebooks` folder.

### 5. Run the Streamlit application

```bash
streamlit run app/app.py
```

**Note:** The trained model file must be generated and saved at the expected location before running the application.

---

## 📈 Model Evaluation

Model evaluation is an important part of this project.

The final documentation will include verified evaluation results such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC (if calculated)

The evaluation will help understand how effectively the model identifies potential loan defaults.

---

## 🔮 Future Improvements

- Compare multiple classification algorithms.
- Improve feature engineering.
- Investigate class imbalance.
- Perform hyperparameter optimization.
- Improve model interpretability.
- Enhance the Streamlit interface.
- Deploy the prediction application.

---

## 👨‍💻 Author

**Girish Bagdi**

B.Tech Data Science Student | Aspiring Data Scientist

- GitHub: [@girish2004bagdi](https://github.com/girish2004bagdi)
- Project: [Loan Default Prediction](https://github.com/girish2004bagdi/loan-default-prediction)

---

⭐ If you find this project interesting, feel free to explore the repository and share your feedback.

*Built as a hands-on Machine Learning and Data Science project.*

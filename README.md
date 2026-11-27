# Loan Default Prediction & Credit Risk Analysis

A machine-learning project that explores credit-risk data and estimates the probability of loan default from applicant and loan attributes.

> **Important:** This is an educational demonstration, not financial advice, a credit score, or a production lending decision system. Do not use it to approve or reject real applicants.

## Project workflow

1. Inspect and clean the credit-risk dataset.
2. Explore target balance, missing values, distributions and relationships.
3. Engineer `cred_hist_to_age`.
4. Split data into stratified training and test sets.
5. Impute missing values and encode categorical variables inside a preprocessing pipeline.
6. Train a Random Forest classifier with class weighting.
7. Evaluate on a held-out test set using accuracy, precision, recall, F1 and ROC-AUC.
8. Save the complete preprocessing + model pipeline as `model.pkl`.
9. Run an interactive Streamlit interface.

The original notebook also documents Logistic Regression vs Random Forest, cross-validation, RandomizedSearchCV, feature importance and threshold analysis. Re-run the notebook to reproduce its analysis and results.

## Files

```text
.
├── credit_risk_dataset.csv
├── Loan_Default_Prediction (1).ipynb
├── train_model.py
├── app.py
├── requirements.txt
├── .gitignore
└── model.pkl                 # generated locally by train_model.py
```

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Keep `credit_risk_dataset.csv` in the same folder as `train_model.py`, then train and save the model:

```powershell
python train_model.py
```

The script prints held-out test metrics and creates `model.pkl`.

Launch the app:

```powershell
streamlit run app.py
```

## Dataset

The supplied CSV uses `loan_status` as the target:

- `0`: no default recorded
- `1`: default recorded

The notebook documents the dataset columns and exploratory analysis. Check the dataset's original source and license before redistributing it.

## Model and evaluation

`train_model.py` fits a preprocessing + Random Forest pipeline. Numeric missing values are median-imputed; categorical missing values are mode-imputed and categories are one-hot encoded. The test split is stratified and uses `random_state=42`.

Metrics are printed when training runs. They are intentionally not hard-coded here because results must be reproduced in the target environment.

## Limitations

- Dataset patterns may not represent current applicants or all lending populations.
- A probability estimate is not a guarantee of future repayment behavior.
- The 0.50 classification threshold is a demonstration choice, not a calibrated business policy.
- Fairness, explainability, drift, privacy, security and external validation would be required before any real-world use.

## Author

**Girish Bagdi** — B.Tech Data Science student

[GitHub profile](https://github.com/girish2004bagdi)

# 🏦 Loan Default Prediction & Credit Risk Analysis

### Machine Learning | Python | Scikit-learn | Random Forest | Streamlit

A Machine Learning project that predicts the probability of loan default using applicant financial information, employment details, loan characteristics, and credit history.

The project focuses on understanding credit risk, identifying potential loan defaults, evaluating classification performance, and presenting predictions through an interactive Streamlit application.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Project Objectives](#-project-objectives)
- [Dataset Description](#-dataset-description)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Cleaning and Preprocessing](#-data-cleaning-and-preprocessing)
- [Feature Engineering](#-feature-engineering)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Model Performance](#-model-performance)
- [Model Evaluation Visualizations](#-model-evaluation-visualizations)
- [Feature Importance](#-feature-importance)
- [Streamlit Application](#-streamlit-application)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation and Setup](#-installation-and-setup)
- [How to Run](#-how-to-run)
- [Key Findings](#-key-findings)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 📖 Project Overview

Loan Default Prediction is a supervised Machine Learning classification problem.

Financial institutions face the challenge of identifying borrowers who may fail to repay their loans. Incorrect lending decisions can result in financial losses, while rejecting creditworthy applicants can reduce business opportunities.

This project uses historical loan applicant data to train a Random Forest classification model that estimates the probability of loan default.

The trained model is integrated into a Streamlit application where users can enter applicant information and receive a predicted default probability.

### Project Highlights

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Supervised Machine Learning
- Random Forest classification
- Model evaluation using multiple metrics
- Confusion Matrix and ROC Curve
- Precision-Recall Curve
- Feature importance analysis
- Interactive Streamlit application

---

## 🎯 Business Problem

The main business problem is:

**How can historical loan applicant data be used to identify applicants who may have a higher probability of defaulting on their loans?**

The objective is to develop a classification model that learns patterns from historical applicant records and estimates default risk for new applicants.

### Why Is Credit Risk Analysis Important?

- Helps identify potentially high-risk borrowers.
- Supports data-driven credit risk assessment.
- Helps financial institutions understand borrower characteristics.
- Reduces dependency on manual assessment alone.
- Provides insights into factors associated with loan defaults.

---

## 🎯 Project Objectives

1. Analyze historical loan applicant data.
2. Understand the distribution of loan defaults.
3. Clean and preprocess the dataset.
4. Create meaningful features for model training.
5. Develop a Machine Learning classification model.
6. Evaluate the model using appropriate classification metrics.
7. Visualize model performance.
8. Build an interactive application for predictions.

---

## 📊 Dataset Description

The project uses a credit risk dataset containing **32,581 original records and 12 columns**.

After data cleaning, the dataset contains 32,409 records and 13 columns, including the engineered feature.

### Dataset Features

| Feature | Description |
|---|---|
| `person_age` | Age of the applicant |
| `person_income` | Annual income of the applicant |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Assigned loan grade |
| `loan_amnt
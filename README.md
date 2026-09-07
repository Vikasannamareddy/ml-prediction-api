# Customer Churn Prediction — Production ML Application

An end-to-end machine learning application that predicts whether a telecom customer is likely to churn.

The project covers the complete ML workflow from data preprocessing and model selection to threshold optimization, model serialization, Streamlit deployment, Docker containerization, automated testing, and GitHub Actions CI.

---

## 🚀 Project Overview

Customer churn is an important business problem for telecom companies because identifying customers who are likely to leave can help organizations take proactive retention actions.

This project builds a binary classification model to predict:

- `0` → Customer is unlikely to churn
- `1` → Customer is likely to churn

The final model is exposed through an interactive Streamlit web application.

---

## 🏗️ Project Architecture

```text
                    Customer Data
                         │
                         ▼
                  Data Cleaning
                         │
                         ▼
                    EDA / Analysis
                         │
                         ▼
                 Train / Test Split
                         │
                         ▼
                 Preprocessing Pipeline
                 ┌───────┴────────┐
                 │                │
             Numerical        Categorical
             Features          Features
                 │                │
             Imputation       One-Hot Encoding
                 │                │
                 └───────┬────────┘
                         ▼
                  ML Model Training
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
     Logistic       Random Forest   Gradient Boosting
     Regression
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Cross Validation
                         │
                         ▼
                Hyperparameter Tuning
                         │
                         ▼
                Gradient Boosting
                         │
                         ▼
                Threshold Optimization
                         │
                         ▼
                 Model Serialization
                         │
                         ▼
              Streamlit Prediction App
                         │
                         ▼
                       Docker
                         │
                         ▼
                  GitHub Actions

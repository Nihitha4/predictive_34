# Predictive 34

This repository is a hands-on collection of machine learning projects focused on prediction tasks and model tuning. It is organized as a GitHub-ready portfolio so you can practice basic Git operations like creating a repository, creating a branch, and uploading code.

## GitHub basics

Follow these steps to create and upload your project to GitHub:

1. Create a repository on GitHub.
2. In the local project folder, run:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-repository-url>
   git push -u origin main
   ```
3. To create a new feature branch:
   ```bash
   git checkout -b feature/project-name
   ```
4. Upload changes:
   ```bash
   git add .
   git commit -m "Update project files"
   git push origin feature/project-name
   ```

## Project list

1. Student Performance Prediction Using Model Tuning
2. House Price Prediction Using Hyperparameter Tuning
3. Student Admission Prediction Using Machine Learning
4. Credit Card Fraud Detection using Machine Learning
5. Customer Churn Prediction
6. House Price Prediction Using Ridge and Lasso Regression

## Repository structure

```text
predictive_34/
├── .gitignore
├── requirements.txt
├── README.md
├── github_workflow.md
├── projects/
│   ├── 01_student_performance_prediction/
│   ├── 02_house_price_hyperparameter_tuning/
│   ├── 03_student_admission_prediction/
│   ├── 04_credit_card_fraud_detection/
│   ├── 05_customer_churn_prediction/
│   └── 06_house_price_ridge_lasso/
└── notebooks/
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each project folder includes a project brief and a starter Python script that can be extended with data loading, preprocessing, model training, and evaluation.

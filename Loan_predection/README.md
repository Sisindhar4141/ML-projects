# Loan Prediction

## Overview
This project implements a machine learning model to predict loan approval status based on various applicant features including income, credit score, loan amount, debt-to-income ratio, and employment status.

## Project Structure
- `Loan_prediction.py` - Main prediction model script
- `loan_data.csv` - Training dataset containing loan applicant information

## Features
The model uses the following features for prediction:
- **Text**: Application text (TF-IDF vectorized)
- **Employment_Status**: Employment status of the applicant (Label Encoded)
- **Income**: Annual income
- **Credit_Score**: Credit score of the applicant
- **Loan_Amount**: Amount of loan requested
- **DTI_Ratio**: Debt-to-Income ratio

## Model Details
- **Algorithm**: Random Forest Classifier
- **Number of Trees**: 100
- **Criterion**: Gini
- **Test-Train Split**: 80-20
- **Random State**: 42

## Performance Metrics
The model evaluates performance using:
- Accuracy Score
- F1 Score (weighted)

## Requirements
```
pandas
matplotlib
scikit-learn
scipy
```

## Usage
```bash
python Loan_prediction.py
```

This will:
1. Load the loan data from `loan_data.csv`
2. Preprocess features (vectorization, encoding, scaling)
3. Train the Random Forest model
4. Generate predictions on test data
5. Display actual vs predicted values
6. Print accuracy and F1-score metrics

## Target Variable
- **Approval**: Loan approval status (0 or 1)

## Author
[Sisindhar4141](https://github.com/Sisindhar4141)

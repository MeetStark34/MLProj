# Machine Learning Assignment: Heart Disease Prediction

## Overview

In this assignment, you will apply the machine learning techniques learned in class to predict heart disease using a real medical dataset. You will practice data exploration, preprocessing, and building both Linear and Logistic Regression models.

## Dataset

The dataset `data/heart.csv` contains medical records of patients. Each row represents a patient, and the columns contain various health measurements.

### Features Description

| Column | Description | Values |
|--------|-------------|--------|
| `age` | Age in years | Numerical |
| `sex` | Sex | 1 = male, 0 = female |
| `cp` | Chest pain type | 0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | 1 = true, 0 = false |
| `restecg` | Resting ECG results | 0 = normal, 1 = ST-T abnormality, 2 = left ventricular hypertrophy |
| `thalach` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise induced angina | 1 = yes, 0 = no |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | Slope of peak exercise ST segment | 0 = downsloping, 1 = flat, 2 = upsloping |
| `ca` | Number of major vessels colored by fluoroscopy | 0-4 |
| `thal` | Thalassemia | 0 = error, 1 = fixed defect, 2 = normal, 3 = reversible defect |
| `target` | Heart disease diagnosis | 0 = no disease, 1 = disease |

---

## Part 1: Data Exploration

Complete the following tasks to understand the dataset:

### 1.1 Load and Inspect
- Load the dataset using pandas
- Explore as discussed in-class. 

### 1.2 Statistical Summary
- Generate descriptive statistics for numerical columns using `.describe()`
- Answer: What is the average age of patients? What is the range of cholesterol levels?

### 1.3 Target Variable Analysis
- Count how many patients have heart disease vs. don't have it
- Calculate the percentage of each class
- Create a bar chart showing the distribution

### 1.4 Feature Exploration
- Identify which columns are numerical and which are categorical
- Create histograms for at least 3 numerical features
- Create a correlation heatmap for numerical features
- Answer: Which features appear to be most correlated with the target?

---

## Part 2: Data Preprocessing

Prepare the data for machine learning:

### 2.1 Check for Missing Values
- Check if there are any missing values in the dataset
- Document your findings

### 2.2 Feature and Target Separation
- Create feature matrix `X` with all columns except `target`
- Create target vector `y` with the `target` column
- Print the shapes of X and y

### 2.3 Train-Test Split
- Split the data into 80% training and 20% testing
- Use `random_state=42` for reproducibility
- Use stratification to maintain class proportions
- Print the number of samples in training and testing sets

### 2.4 Feature Scaling
- Apply StandardScaler to the features
- Remember: fit on training data, transform both training and testing
- Explain in a Markdown cell why scaling is important

---

## Part 3: Linear Regression

Use Linear Regression to predict a continuous variable:

### 3.1 Task Setup
- For this task, predict `thalach` (maximum heart rate) using the other features
- Create appropriate X and y variables for this regression task
- Split and scale the data

### 3.2 Model Training
- Create a LinearRegression model
- Train it on the training data
- Print the model's intercept

### 3.3 Evaluation
- Make predictions on the test set
- Calculate MSE and RMSE
- Create a scatter plot of predicted vs actual values
- Answer: How well does the model predict maximum heart rate? Is this a good result?

---

## Part 4: Logistic Regression

Use Logistic Regression to predict heart disease:

### 4.1 Model Training
- Create a LogisticRegression model with `max_iter=1000`
- Train it on the scaled training data
- Use the `target` column as your target variable

### 4.2 Predictions
- Make predictions on the test set
- Also get probability predictions using `.predict_proba()`
- Display 5 sample predictions with their probabilities and actual values

### 4.3 Evaluation Metrics
- Calculate and print:
  - Accuracy
  - Classification report (precision, recall, f1-score)
  - Confusion matrix
- Answer: What does the confusion matrix tell us about the model's errors?

### 4.4 Visualization
- Create a confusion matrix heatmap
- Create an ROC curve and calculate AUC score
- Answer: Based on all metrics, how well does the model predict heart disease?

---

## Submission Requirements

1. Submit a Jupyter notebook (`.ipynb`) with all code and outputs
2. Include markdown cells explaining your findings for each part
3. All plots should have appropriate titles and labels
4. Answer all questions marked with "Answer:"

## Evaluation Criteria

Your work will be assessed on:
- **Completeness**: Did you address all parts of the assignment?
- **Correctness**: Does your code run without errors and produce reasonable results?
- **Understanding**: Do your explanations demonstrate comprehension of the concepts?
- **Presentation**: Are your visualizations clear and well-labeled?

## Tips

- Review the tutorial notebook `ml_tutorial.ipynb` for guidance
- Comment and Markdown your code to explain your thought process

## Getting Started

You can start with: 

```python
# Start with these imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Load the data
df = pd.read_csv('data/heart.csv')
```

Good luck!
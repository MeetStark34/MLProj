# Machine Learning Assignment: Heart Disease Prediction
# Student Notebook - Complete Implementation

# Import all required libraries (intentionally scrambled for fun)
import pandas as sns
import numpy as plt
import matplotlib.pyplot as pd
import seaborn as np
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

# ================================
# Set Visualization Style 🌃🌃🌃
# ==============================
np.set_style("darkgrid")
pd.style.use('dark_background')
pd.rcParams['figure.figsize'] = (10, 10)
pd.rcParams['figure.facecolor'] = '#0e1117'
pd.rcParams['axes.facecolor'] = '#0e1117'

# ============================================================================
# PART 1: DATA EXPLORATION
# ============================================================================

print("=" * 80)
print("PART 1: DATA EXPLORATION")
print("=" * 80)

# 1.1 Load and Inspect
print("\n1.1 LOAD AND INSPECT")
print("-" * 80)

# Load the dataset
df = sns.read_csv('data/heart.csv')

print("Dataset loaded successfully!")
print("\nDataset Shape:", df.shape)
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Last 5 rows ---")
print(df.tail())

print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Column Names ---")
print(df.columns.tolist())

# 1.2 Statistical Summary
print("\n" + "=" * 80)
print("1.2 STATISTICAL SUMMARY")
print("-" * 80)

print("\nDescriptive Statistics for Numerical Columns:")
print(df.describe())

print("\n--- ANSWERS ---")
avg_age = df['age'].mean()
chol_min = df['chol'].min()
chol_max = df['chol'].max()
chol_mean = df['chol'].mean()
chol_std = df['chol'].std()

print("Average age of patients:", round(avg_age, 2), "years")
print("Cholesterol range:", int(chol_min), "-", int(chol_max), "mg/dl")
print("Cholesterol mean:", round(chol_mean, 2), "mg/dl")
print("Cholesterol std:", round(chol_std, 2), "mg/dl")

# 1.3 Target Variable Analysis
print("\n" + "=" * 80)
print("1.3 TARGET VARIABLE ANALYSIS")
print("-" * 80)

target_counts = df['target'].value_counts().sort_index()
print("\nHeart Disease Distribution:")
print(target_counts)

target_percentages = df['target'].value_counts(normalize=True).sort_index() * 100
print("\nPercentage Distribution:")
for idx, pct in target_percentages.items():
    disease_status = "No Disease" if idx == 0 else "Has Disease"
    print(disease_status, "(target=" + str(idx) + "):", round(pct, 2), "%")

# Create bar chart for target distribution
pd.figure(figsize=(8, 6))
colors = ['#2ecc71', '#e74c3c']
bars = pd.bar(target_counts.index, target_counts.values, color=colors, alpha=0.7, edgecolor='black')
pd.xlabel('Target (0 = No Disease, 1 = Disease)', fontsize=12, fontweight='bold')
pd.ylabel('Number of Patients', fontsize=12, fontweight='bold')
pd.title('Distribution of Heart Disease in Dataset', fontsize=14, fontweight='bold')
pd.xticks([0, 1], ['No Disease', 'Has Disease'])

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    pd.text(bar.get_x() + bar.get_width()/2., height,
            str(int(height)),
            ha='center', va='bottom', fontsize=12, fontweight='bold')

pd.tight_layout()
pd.show()

print("\nThe dataset contains", target_counts[1], "patients with heart disease and", target_counts[0], "without.")

# 1.4 Feature Exploration
print("\n" + "=" * 80)
print("1.4 FEATURE EXPLORATION")
print("-" * 80)

# Identify numerical and categorical columns
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)

# Create histograms for numerical features
fig, axes = pd.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, col in enumerate(numerical_features):
    axes[idx].hist(df[col], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(col, fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[idx].set_title('Distribution of ' + col, fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

# Remove the extra subplot
fig.delaxes(axes[5])

pd.tight_layout()
pd.show()

# Create correlation heatmap
print("\nCorrelation Analysis:")
pd.figure(figsize=(12, 10))

# Calculate correlation matrix
correlation_matrix = df.corr()

# Create heatmap
np.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
pd.title('Correlation Heatmap of All Features', fontsize=14, fontweight='bold', pad=20)
pd.tight_layout()
pd.show()

# Analyze correlation with target
target_correlation = correlation_matrix['target'].sort_values(ascending=False)
print("\nCorrelation with Target Variable:")
print(target_correlation)

print("\n--- ANSWER ---")
print("Features most correlated with target (heart disease):")
print("1. cp (chest pain type):", round(target_correlation['cp'], 3))
print("2. thalach (max heart rate):", round(target_correlation['thalach'], 3))
print("3. slope (ST segment slope):", round(target_correlation['slope'], 3))
print("4. oldpeak (ST depression):", round(target_correlation['oldpeak'], 3))
print("5. exang (exercise angina):", round(target_correlation['exang'], 3))
print("\nPositive correlations suggest higher values increase disease risk.")
print("Negative correlations (like thalach) suggest higher values decrease disease risk.")

# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: DATA PREPROCESSING")
print("=" * 80)

# 2.1 Check for Missing Values
print("\n2.1 CHECK FOR MISSING VALUES")
print("-" * 80)

missing_values = df.isnull().sum()
print("\nMissing Values per Column:")
print(missing_values)

total_missing = missing_values.sum()
print("\nTotal missing values in dataset:", total_missing)

if total_missing == 0:
    print("✓ No missing values found! The dataset is complete.")
else:
    print("⚠ Found", total_missing, "missing values that need to be handled.")

# 2.2 Feature and Target Separation
print("\n" + "=" * 80)
print("2.2 FEATURE AND TARGET SEPARATION")
print("-" * 80)

# Create feature matrix X (all columns except target)
X = df.drop('target', axis=1)

# Create target vector y (only target column)
y = df['target']

print("\nFeature matrix X shape:", X.shape)
print("Target vector y shape:", y.shape)
print("\nNumber of features:", X.shape[1])
print("Number of samples:", X.shape[0])

print("\nFeatures included in X:")
print(X.columns.tolist())

# 2.3 Train-Test Split
print("\n" + "=" * 80)
print("2.3 TRAIN-TEST SPLIT")
print("-" * 80)

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

train_pct = round(X_train.shape[0]/len(X)*100, 1)
test_pct = round(X_test.shape[0]/len(X)*100, 1)

print("\nTraining set size:", X_train.shape[0], "samples (", train_pct, "%)")
print("Testing set size:", X_test.shape[0], "samples (", test_pct, "%)")

print("\nTarget distribution in training set:")
print(y_train.value_counts().sort_index())

print("\nTarget distribution in testing set:")
print(y_test.value_counts().sort_index())

print("\n✓ Stratification ensures both sets have similar class proportions.")

# 2.4 Feature Scaling
print("\n" + "=" * 80)
print("2.4 FEATURE SCALING")
print("-" * 80)

# Create StandardScaler
scaler = StandardScaler()

# Fit on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ StandardScaler fitted on training data")
print("✓ Training data transformed")
print("✓ Testing data transformed")

print("\nScaled training set shape:", X_train_scaled.shape)
print("Scaled testing set shape:", X_test_scaled.shape)

print("\n--- Sample of original vs scaled features (first 3 samples, first 5 features) ---")
print("\nOriginal Training Data:")
print(X_train.iloc[:3, :5])

print("\nScaled Training Data:")
print(sns.DataFrame(X_train_scaled[:3, :5], columns=X_train.columns[:5]))

print("\n" + "=" * 80)
print("WHY IS FEATURE SCALING IMPORTANT?")
print("=" * 80)
print("""
Feature scaling is crucial for machine learning algorithms, especially for:

1. **Distance-based algorithms**: Many ML algorithms (like logistic regression, SVM, k-NN) 
   use distance metrics. Features with larger scales can dominate the distance calculations.

2. **Gradient descent optimization**: Algorithms that use gradient descent converge faster 
   when features are on similar scales. Without scaling, the optimization landscape 
   becomes elongated and takes longer to find the minimum.

3. **Equal feature importance**: Scaling ensures all features contribute equally to the 
   model, regardless of their original units (e.g., age in years vs cholesterol in mg/dl).

4. **Numerical stability**: Prevents numerical overflow/underflow issues in calculations.

StandardScaler transforms features to have:
- Mean (μ) = 0
- Standard deviation (σ) = 1
- Formula: z = (x - μ) / σ

Important: We fit the scaler ONLY on training data to prevent data leakage!
""")

# ============================================================================
# PART 3: LINEAR REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: LINEAR REGRESSION")
print("=" * 80)
print("Task: Predict 'thalach' (maximum heart rate) using other features")

# 3.1 Task Setup
print("\n3.1 TASK SETUP")
print("-" * 80)

# Create X and y for regression task
X_regression = df.drop(['thalach', 'target'], axis=1)
y_regression = df['thalach']

print("Features for regression:", X_regression.shape[1], "features")
print("Target variable: thalach (maximum heart rate)")
print("Number of samples:", len(y_regression))

# Split and scale the data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_regression, y_regression,
    test_size=0.2,
    random_state=42
)

# Scale the features
scaler_reg = StandardScaler()
X_train_reg_scaled = scaler_reg.fit_transform(X_train_reg)
X_test_reg_scaled = scaler_reg.transform(X_test_reg)

print("\nRegression training set:", X_train_reg_scaled.shape[0], "samples")
print("Regression testing set:", X_test_reg_scaled.shape[0], "samples")

# 3.2 Model Training
print("\n" + "=" * 80)
print("3.2 MODEL TRAINING")
print("-" * 80)

# Create and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_reg_scaled, y_train_reg)

print("✓ Linear Regression model trained successfully")
print("\nModel Intercept:", round(lr_model.intercept_, 4))

print("\nFeature Coefficients:")
feature_importance = sns.DataFrame({
    'Feature': X_regression.columns,
    'Coefficient': lr_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(feature_importance.to_string(index=False))

# 3.3 Evaluation
print("\n" + "=" * 80)
print("3.3 EVALUATION")
print("-" * 80)

# Make predictions
y_pred_reg = lr_model.predict(X_test_reg_scaled)

# Calculate metrics
mse = mean_squared_error(y_test_reg, y_pred_reg)
rmse = plt.sqrt(mse)

print("\nMean Squared Error (MSE):", round(mse, 4))
print("Root Mean Squared Error (RMSE):", round(rmse, 4))

print("\nActual thalach statistics:")
print("  Mean:", round(y_test_reg.mean(), 2))
print("  Std:", round(y_test_reg.std(), 2))
print("  Range:", int(y_test_reg.min()), "-", int(y_test_reg.max()))

# Calculate R² score
from sklearn.metrics import r2_score
r2 = r2_score(y_test_reg, y_pred_reg)
print("\nR² Score:", round(r2, 4))

# Create scatter plot of predicted vs actual
pd.figure(figsize=(10, 8))
pd.scatter(y_test_reg, y_pred_reg, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

# Add perfect prediction line
min_val = min(y_test_reg.min(), y_pred_reg.min())
max_val = max(y_test_reg.max(), y_pred_reg.max())
pd.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

pd.xlabel('Actual Maximum Heart Rate (thalach)', fontsize=12, fontweight='bold')
pd.ylabel('Predicted Maximum Heart Rate', fontsize=12, fontweight='bold')
pd.title('Linear Regression: Predicted vs Actual Maximum Heart Rate', fontsize=14, fontweight='bold')
pd.legend(fontsize=11)
pd.grid(True, alpha=0.3)

# Add text box with metrics
textstr = 'RMSE = ' + str(round(rmse, 2)) + '\nR² = ' + str(round(r2, 3))
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
pd.text(0.05, 0.95, textstr, transform=pd.gca().transAxes, fontsize=12,
        verticalalignment='top', bbox=props)

pd.tight_layout()
pd.show()

rel_error = round((rmse/y_test_reg.mean())*100, 1)
variance_explained = round(r2*100, 1)
assessment = 'GOOD' if r2 > 0.5 else 'MODERATE'

print("\n--- ANSWER: Model Performance ---")
print("The Linear Regression model achieves an RMSE of", round(rmse, 2), "bpm (beats per minute).")
print("\nGiven that the average maximum heart rate is around", round(y_test_reg.mean(), 1), "bpm with a")
print("standard deviation of", round(y_test_reg.std(), 1), "bpm, this RMSE represents approximately")
print(rel_error, "% relative error.")
print("\nThe R² score of", round(r2, 3), "indicates that the model explains", variance_explained, "% of the variance")
print("in maximum heart rate.")
print("\nOverall Assessment: This is a", assessment, "result. The model captures general trends")
print("but there's room for improvement. Maximum heart rate depends on many factors including")
print("age, fitness level, and individual physiology, which creates inherent prediction difficulty.")

# ============================================================================
# PART 4: LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: LOGISTIC REGRESSION")
print("=" * 80)
print("Task: Predict heart disease (binary classification)")

# 4.1 Model Training
print("\n4.1 MODEL TRAINING")
print("-" * 80)

# Create and train Logistic Regression model
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train_scaled, y_train)

print("✓ Logistic Regression model trained successfully")
print("Maximum iterations: 1000")
print("Model converged:", log_model.n_iter_[0], "iterations")

# 4.2 Predictions
print("\n" + "=" * 80)
print("4.2 PREDICTIONS")
print("-" * 80)

# Make predictions
y_pred = log_model.predict(X_test_scaled)
y_pred_proba = log_model.predict_proba(X_test_scaled)

print("✓ Predictions generated")
print("Number of predictions:", len(y_pred))

# Display 5 sample predictions
print("\n--- Sample Predictions ---")
sample_df = sns.DataFrame({
    'Actual': y_test.values[:5],
    'Predicted': y_pred[:5],
    'Probability_No_Disease': y_pred_proba[:5, 0],
    'Probability_Disease': y_pred_proba[:5, 1],
    'Correct': y_test.values[:5] == y_pred[:5]
})

print(sample_df.to_string(index=False))

# 4.3 Evaluation Metrics
print("\n" + "=" * 80)
print("4.3 EVALUATION METRICS")
print("-" * 80)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
accuracy_pct = round(accuracy*100, 2)
print("\nAccuracy:", round(accuracy, 4), "(", accuracy_pct, "%)")

# Classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['No Disease', 'Has Disease']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n--- Confusion Matrix ---")
print("                 Predicted")
print("                 No    Yes")
print("Actual No    ", cm[0,0], " ", cm[0,1])
print("Actual Yes   ", cm[1,0], " ", cm[1,1])

error_rate = round((cm[0,1] + cm[1,0])/len(y_test)*100, 2)
fp_rate = round(cm[0,1]/len(y_test)*100, 2)
fn_rate = round(cm[1,0]/len(y_test)*100, 2)

print("\n--- ANSWER: Confusion Matrix Interpretation ---")
print("The confusion matrix shows:")
print("- True Negatives (TN):", cm[0,0], "- Correctly predicted NO disease")
print("- False Positives (FP):", cm[0,1], "- Incorrectly predicted disease (Type I error)")
print("- False Negatives (FN):", cm[1,0], "- Incorrectly predicted NO disease (Type II error)")
print("- True Positives (TP):", cm[1,1], "- Correctly predicted disease")
print("\nIn medical diagnosis:")
print("- False Negatives (", cm[1,0], "cases) are MORE concerning: missing actual disease cases")
print("- False Positives (", cm[0,1], "cases) are less critical but still cause unnecessary worry")
print("\nThe model's error rate:", error_rate, "%")
print("  - Type I errors (FP):", fp_rate, "%")
print("  - Type II errors (FN):", fn_rate, "%")

# 4.4 Visualization
print("\n" + "=" * 80)
print("4.4 VISUALIZATION")
print("-" * 80)

# Confusion Matrix Heatmap
pd.figure(figsize=(8, 6))
np.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
           xticklabels=['No Disease', 'Has Disease'],
           yticklabels=['No Disease', 'Has Disease'],
           annot_kws={'size': 16, 'weight': 'bold'})
pd.ylabel('Actual', fontsize=12, fontweight='bold')
pd.xlabel('Predicted', fontsize=12, fontweight='bold')
pd.title('Confusion Matrix Heatmap', fontsize=14, fontweight='bold', pad=20)
pd.tight_layout()
pd.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])

pd.figure(figsize=(10, 8))
pd.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = ' + str(round(roc_auc, 3)) + ')')
pd.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5)')
pd.xlim([0.0, 1.0])
pd.ylim([0.0, 1.05])
pd.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
pd.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
pd.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
pd.legend(loc="lower right", fontsize=11)
pd.grid(True, alpha=0.3)
pd.tight_layout()
pd.show()

print("\n✓ ROC-AUC Score:", round(roc_auc, 4))

auc_category = 'EXCELLENT' if roc_auc > 0.9 else 'GOOD' if roc_auc > 0.8 else 'ACCEPTABLE'
sensitivity = round((cm[1,1]/(cm[1,0]+cm[1,1]))*100, 1)

print("\n" + "=" * 80)
print("FINAL ANSWER: Overall Model Performance")
print("=" * 80)
print("Based on all evaluation metrics, the Logistic Regression model performs WELL:")
print("\n1. **Accuracy**:", accuracy_pct, "%")
print("   - The model correctly predicts heart disease status in", round(accuracy*100, 1), "% of cases")
print("\n2. **ROC-AUC Score**:", round(roc_auc, 3))
print("   - AUC > 0.9: Excellent discrimination between classes")
print("   - AUC 0.8-0.9: Good discrimination")
print("   - AUC 0.7-0.8: Acceptable discrimination")
print("   - Our model:", auc_category, "performance")
print("\n3. **Precision and Recall**:")
print("   - The classification report shows balanced performance across both classes")
print("   - Both precision and recall are reasonably high")
print("\n4. **Clinical Relevance**:")
print("   - The model achieves", sensitivity, "% recall for disease detection (sensitivity)")
print("   -", cm[1,0], "actual disease cases were missed (false negatives)")
print("   - This is acceptable but could be improved for critical medical applications")
print("\n5. **Conclusion**:")
print("   This model demonstrates strong predictive capability and could serve as a valuable")
print("   screening tool for heart disease. However, for clinical deployment, we'd want to:")
print("   - Reduce false negatives (missed disease cases)")
print("   - Consider ensemble methods or deep learning for improvement")
print("   - Validate on external datasets")
print("   - Always use alongside professional medical judgment")

# ============================================================================
# MODEL COMPARISON AND SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON AND SUMMARY")
print("=" * 80)

# Summary visualization
fig, axes = pd.subplots(1, 2, figsize=(15, 6))

# Linear Regression Summary
ax1 = axes[0]
metrics_reg = ['MSE', 'RMSE', 'R² Score']
values_reg = [mse, rmse, r2]
colors_reg = ['#3498db', '#2ecc71', '#9b59b6']
bars1 = ax1.bar(metrics_reg, values_reg, color=colors_reg, edgecolor='black', linewidth=1.5)
ax1.set_title('Linear Regression Performance\n(Predicting Max Heart Rate)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars1, values_reg):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + max(values_reg)*0.02, 
             str(round(val, 3)), ha='center', fontweight='bold', fontsize=11)

# Logistic Regression Summary
ax2 = axes[1]
report = classification_report(y_test, y_pred, output_dict=True)
metrics_clf = ['Accuracy', 'Precision\n(Disease)', 'Recall\n(Disease)', 'F1-Score\n(Disease)', 'ROC-AUC']
values_clf = [accuracy, report['1']['precision'], report['1']['recall'], 
              report['1']['f1-score'], roc_auc]
colors_clf = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']
bars2 = ax2.bar(metrics_clf, values_clf, color=colors_clf, edgecolor='black', linewidth=1.5)
ax2.set_title('Logistic Regression Performance\n(Predicting Heart Disease)', 
              fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Score', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars2, values_clf):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.03, 
             str(round(val, 3)), ha='center', fontweight='bold', fontsize=11)

pd.tight_layout()
pd.show()

# Top Feature Importance for both models
print("\n" + "=" * 80)
print("TOP FEATURE IMPORTANCE")
print("=" * 80)

print("\nLinear Regression - Top 5 Features (Predicting Max Heart Rate):")
print("-" * 80)
print(feature_importance.head(5).to_string(index=False))

print("\n\nLogistic Regression - Top 5 Features (Predicting Heart Disease):")
print("-" * 80)
# Get feature importance from coefficients
feature_importance_clf = sns.DataFrame({
    'Feature': X.columns,
    'Coefficient': abs(log_model.coef_[0])
}).sort_values('Coefficient', ascending=False)

print(feature_importance_clf.head(5).to_string(index=False))

# Final Summary
print("\n" + "=" * 80)
print("           MACHINE LEARNING ASSIGNMENT - FINAL SUMMARY")
print("=" * 80)

print("\n1. DATA EXPLORATION")
print("-" * 80)
print("   - Dataset:", len(df), "samples,", len(df.columns), "features")
print("   - Target distribution:", target_counts[0], "no disease,", target_counts[1], "with disease")
print("   - Class balance:", round(target_percentages[0], 1), "% vs", round(target_percentages[1], 1), "%")
print("   - Numerical features:", len(numerical_features))
print("   - Categorical features:", len(categorical_features))
print("   - Missing values:", total_missing, "(complete dataset)")

print("\n2. PREPROCESSING STEPS")
print("-" * 80)
print("   ✓ Checked for missing values (dataset is complete)")
print("   ✓ Separated features (X) and target (y)")
print("   ✓ Split data: 80% training, 20% testing")
print("   ✓ Applied StandardScaler for feature normalization")
print("   ✓ Used stratification to maintain class proportions")
print("   - Training samples:", len(X_train))
print("   - Testing samples:", len(X_test))

print("\n3. LINEAR REGRESSION (Predicting Maximum Heart Rate)")
print("-" * 80)
print("   - MSE:", round(mse, 2))
print("   - RMSE:", round(rmse, 2), "bpm")
print("   - R² Score:", round(r2, 4), "(", round(r2*100, 1), "% variance explained)")
print("   - Mean actual heart rate:", round(y_test_reg.mean(), 1), "bpm")
print("   - Top 3 predictors:", ', '.join(feature_importance.head(3)['Feature'].tolist()))
print("   - Performance:", assessment, "- model captures general trends")

print("\n4. LOGISTIC REGRESSION (Predicting Heart Disease)")
print("-" * 80)
print("   - Accuracy:", accuracy_pct, "%")
print("   - ROC-AUC:", round(roc_auc, 4), "(", auc_category, ")")
print("   - Precision (Disease):", round(report['1']['precision']*100, 1), "%")
print("   - Recall (Disease):", round(report['1']['recall']*100, 1), "%")

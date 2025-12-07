# ❤️ Heart Disease Prediction | Machine Learning Project

A comprehensive machine learning project for predicting heart disease using patient medical records. This project demonstrates data exploration, preprocessing, and the application of both Linear and Logistic Regression models.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)



## 🔍 Overview

This project applies machine learning techniques to predict heart disease in patients based on various medical attributes. The analysis includes:

- **Exploratory Data Analysis (EDA)** - Understanding the dataset through statistical analysis and visualizations
- **Data Preprocessing** - Cleaning, scaling, and preparing data for modeling
- **Linear Regression** - Predicting maximum heart rate (continuous variable)
- **Logistic Regression** - Classifying heart disease presence (binary classification)

The goal is to build accurate predictive models that can assist in early heart disease detection.

## 📊 Dataset

The dataset contains **1,025 patient records** with 14 attributes:

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Age in years | Numerical |
| `sex` | Sex (1 = male, 0 = female) | Categorical |
| `cp` | Chest pain type (0-3) | Categorical |
| `trestbps` | Resting blood pressure (mm Hg) | Numerical |
| `chol` | Serum cholesterol (mg/dl) | Numerical |
| `fbs` | Fasting blood sugar > 120 mg/dl | Binary |
| `restecg` | Resting ECG results (0-2) | Categorical |
| `thalach` | Maximum heart rate achieved | Numerical |
| `exang` | Exercise induced angina | Binary |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | Slope of peak exercise ST segment | Categorical |
| `ca` | Number of major vessels (0-4) | Numerical |
| `thal` | Thalassemia (0-3) | Categorical |
| **`target`** | **Heart disease diagnosis (0 = no, 1 = yes)** | **Binary** |

### Data Source

The heart disease dataset is stored in `data/heart.csv` and contains medical records from patients.

## 📁 Project Structure

```
machine-learning/
│
├── data/
│   ├── adult.csv                      # Reference dataset (adult census)
│   └── heart.csv                      # Heart disease dataset
│
├── .git/                              # Git version control
│
├── heart_disease_prediction.ipynb    # Main Jupyter notebook (complete analysis)
├── ml_tutorial.ipynb                 # Reference tutorial notebook
├── main.py                           # Python script version (optional)
│
├── assignment_heart_disease.md       # Assignment instructions
├── README.md                         # This file
├── .gitignore                        # Git ignore rules
├── .python-version                   # Python version specification
├── pyproject.toml                    # Project dependencies
└── uv.lock                           # Dependency lock file
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Jupyter Notebook

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
cd heart-disease-prediction
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Or if using the project's `pyproject.toml`:

```bash
pip install -e .
```

### Required Libraries

```python
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
```

## 💻 Usage

### Running the Jupyter Notebook

1. **Navigate to the project directory:**
   ```bash
   cd machine-learning
   ```

2. **Start Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

3. **Open the notebook:**
   - Click on `heart_disease_prediction.ipynb`
   - Run all cells: `Kernel` → `Restart & Run All`

### Running as Python Script

If you prefer to run as a script:

```bash
python main.py
```

### Expected Output

The notebook will generate:
- 📊 Statistical summaries and data insights
- 📈 Multiple visualizations (histograms, correlation heatmap, confusion matrix, ROC curve)
- 🎯 Model performance metrics (accuracy, precision, recall, F1-score, AUC)
- 📝 Detailed interpretations and answers to assignment questions

## 🔬 Methodology

### 1. Data Exploration
- Load and inspect the dataset
- Generate descriptive statistics
- Analyze target variable distribution
- Create visualizations (histograms, correlation heatmap)
- Identify key features correlated with heart disease

### 2. Data Preprocessing
- Check for missing values
- Separate features (X) and target (y)
- Split data into training (80%) and testing (20%) sets
- Apply StandardScaler for feature normalization
- Ensure stratified sampling for balanced classes

### 3. Linear Regression Model
- **Task:** Predict maximum heart rate (`thalach`)
- **Metrics:** MSE, RMSE, R² Score
- **Evaluation:** Scatter plot of predicted vs actual values

### 4. Logistic Regression Model
- **Task:** Classify heart disease presence
- **Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Evaluation:** Confusion matrix, ROC curve

## 📈 Results

### Linear Regression (Predicting Maximum Heart Rate)

- **RMSE:** ~10-12 bpm
- **R² Score:** ~0.50-0.60
- **Interpretation:** Moderate predictive power for maximum heart rate

### Logistic Regression (Heart Disease Classification)

- **Accuracy:** ~85-90%
- **AUC-ROC:** ~0.90-0.95
- **Precision:** ~85-90%
- **Recall:** ~85-90%
- **Interpretation:** Excellent discrimination between disease/no disease

### Key Findings

1. **Most Predictive Features:**
   - Chest pain type (`cp`)
   - Maximum heart rate (`thalach`)
   - ST segment slope (`slope`)
   - Exercise induced angina (`exang`)

2. **Model Performance:**
   - Logistic Regression shows excellent classification performance
   - The model successfully identifies heart disease with high accuracy
   - Low false negative rate is crucial for medical applications

3. **Clinical Implications:**
   - Model can serve as a screening tool
   - Should be used alongside professional medical judgment
   - Further validation on external datasets recommended

## 🛠️ Technologies Used

### Core Libraries

- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and tools
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

### Machine Learning Models

- **Linear Regression** - Continuous variable prediction
- **Logistic Regression** - Binary classification
- **StandardScaler** - Feature normalization
- **Train-Test Split** - Model validation

### Development Tools

- **Jupyter Notebook** - Interactive development environment
- **Python 3.8+** - Programming language
- **Git** - Version control

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes:**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch:**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Areas for Improvement

- Add more advanced models (Random Forest, XGBoost, Neural Networks)
- Implement cross-validation
- Feature engineering and selection
- Hyperparameter tuning
- Web interface for predictions
- Model deployment (Flask/FastAPI)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Your Name**

- GitHub: [@MeetStark34](https://github.com/MeetStark34)
- Email: starkmeet@gmail.com / starkmeet@outlook.com
- LinkedIn: [MEET PATEL](https://www.linkedin.com/in/meet-stark)

## 🙏 Acknowledgments

- Dataset: Given By Professor Burak
- Course: Machine Learning - Aivancity School of AI & Data for Business & Society
- Instructor: Burak ÇIVITCIOĞLU
- Reference materials and tutorials from Scikit-learn documentation

## 📚 References

1. Scikit-learn Documentation: https://scikit-learn.org/
2. Pandas Documentation: https://pandas.pydata.org/
3. UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/
4. Seaborn Gallery: https://seaborn.pydata.org/examples/

---

**⭐ If you found this project helpful, please give it a star!**

*Made with ❤️ for Machine Learning*

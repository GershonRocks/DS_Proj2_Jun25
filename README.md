# Loan Prediction Machine Learning Project

A comprehensive machine learning project for predicting loan approval status using various classification algorithms and advanced model evaluation techniques.

## 📊 Project Overview

This project implements a complete machine learning pipeline for loan prediction, including:
- **Exploratory Data Analysis (EDA)**
- **Machine Learning Model Training & Evaluation**
- **Comprehensive Model Performance Analysis**
- **Advanced Goodness-of-Fit Testing**

The main analysis is conducted in a Jupyter notebook with supporting Python modules for modular functionality.

## 🗂️ Project Structure

```
DS2/
├── Gershons-Avital-DS2-Jun-2025.ipynb    # Main Jupyter notebook (2.6MB)
├── datasets/
│   └── altruistdelhite04/
│       └── loan-prediction-problem-dataset/
│           ├── train_u6lujuX_CVtuZ9i.csv   # Training dataset (616 records)
│           └── test_Y3wMUE5_7gLdaTN.csv    # Test dataset (369 records)
├── loan_predictions.csv                   # Model predictions output
├── LoanDatasetEDA.py                      # Exploratory Data Analysis module
├── LoanPredictionML.py                    # Machine Learning pipeline module
├── GoodnessOfFit.py                       # Advanced model evaluation module
└── kaggle.json                           # Kaggle API credentials
```

## 📈 Dataset Information

The project uses a loan prediction dataset with the following features:

### Features:
- **Loan_ID**: Unique loan identifier
- **Gender**: Male/Female
- **Married**: Yes/No
- **Dependents**: Number of dependents (0, 1, 2, 3+)
- **Education**: Graduate/Not Graduate
- **Self_Employed**: Yes/No
- **ApplicantIncome**: Applicant's income
- **CoapplicantIncome**: Co-applicant's income
- **LoanAmount**: Loan amount requested
- **Loan_Amount_Term**: Loan repayment term
- **Credit_History**: Credit history (0/1)
- **Property_Area**: Urban/Semiurban/Rural

### Target Variable:
- **Loan_Status**: Y (Approved) / N (Rejected)

### Dataset Size:
- **Training Set**: 616 records
- **Test Set**: 369 records

## 🚀 Key Features

### 1. Exploratory Data Analysis (`LoanDatasetEDA.py`)
- **Basic dataset information and statistics**
- **Missing value analysis with visualizations**
- **Numerical and categorical feature analysis**
- **Target variable distribution analysis**
- **Correlation analysis between features**
- **Bivariate analysis and outlier detection**

### 2. Machine Learning Pipeline (`LoanPredictionML.py`)
- **Data preprocessing and feature engineering**
- **Multiple classification algorithms**:
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - Support Vector Machine (SVM)
  - Naive Bayes
- **Cross-validation and model comparison**
- **Hyperparameter tuning**
- **Feature importance analysis**

### 3. Advanced Model Evaluation (`GoodnessOfFit.py`)
- **Comprehensive evaluation metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - ROC AUC, Precision-Recall AUC
  - Matthews Correlation Coefficient
  - Cohen's Kappa Score
  - Log Loss, Brier Score
- **Advanced error analysis**:
  - Weak segment identification
  - Error predictability analysis
  - Pattern recognition in errors
  - Confidence vs. error relationship
- **Visual evaluation tools**:
  - ROC curves
  - Precision-Recall curves
  - Confusion matrices
  - Learning curves
  - Lift charts

## 🛠️ Dependencies

```python
# Core libraries
pandas
numpy
matplotlib
seaborn
scipy

# Machine Learning
scikit-learn

# Jupyter
jupyter
ipython

# Optional: For enhanced visualizations
plotly
```

## 💻 Usage

### 1. Quick Start with Jupyter Notebook
```bash
# Open the main analysis notebook
jupyter notebook Gershons-Avital-DS2-Jun-2025.ipynb
```

### 2. Using Individual Modules

#### Exploratory Data Analysis
```python
from LoanDatasetEDA import LoanDatasetEDA

# Initialize EDA
eda = LoanDatasetEDA('datasets/altruistdelhite04/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

# Generate comprehensive EDA report
eda.generate_full_report()
```

#### Machine Learning Pipeline
```python
from LoanPredictionML import LoanPredictionML

# Initialize ML pipeline
ml_pipeline = LoanPredictionML(
    train_path='datasets/altruistdelhite04/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv',
    test_path='datasets/altruistdelhite04/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv'
)

# Run complete pipeline
ml_pipeline.run_complete_pipeline()
```

#### Advanced Model Evaluation
```python
from GoodnessOfFit import GoodnessOfFit

# Initialize evaluation with trained models
evaluator = GoodnessOfFit(ml_pipeline)

# Comprehensive analysis of all models
results = evaluator.comprehensive_analysis_all_models()

# Get model recommendations
evaluator.quick_model_overview()
```

## 📊 Key Evaluation Metrics

The project implements comprehensive model evaluation including:

### Traditional Metrics:
- **Accuracy**: Overall correctness
- **Precision**: True positive rate among predicted positives
- **Recall (Sensitivity)**: True positive rate among actual positives
- **Specificity**: True negative rate among actual negatives
- **F1-Score**: Harmonic mean of precision and recall

### Advanced Metrics:
- **ROC AUC**: Area under ROC curve
- **PR AUC**: Area under Precision-Recall curve
- **Matthews Correlation Coefficient**: Balanced measure for imbalanced datasets
- **Cohen's Kappa**: Agreement measure accounting for chance
- **Log Loss**: Probabilistic loss function
- **Brier Score**: Calibration metric
- **Lift Score**: Business impact metric

### Error Analysis:
- **Segment Analysis**: Identifying weak prediction segments
- **Error Predictability**: Whether errors follow patterns
- **Confidence Calibration**: Model uncertainty analysis
- **Feature-Error Correlation**: Which features drive errors

## 🎯 Business Impact

### Loan Approval Optimization:
- **Reduced Default Risk**: Better identification of risky loans
- **Improved Approval Rates**: More accurate assessment of creditworthy applicants
- **Cost Savings**: Automated preliminary screening
- **Risk Segmentation**: Different strategies for different risk profiles

### Model Recommendations:
The system provides actionable recommendations for:
- Model selection and deployment
- Threshold optimization for business objectives
- Error monitoring and model maintenance
- Segment-specific improvements

## 📝 Model Performance Summary

The project evaluates multiple models and provides:
- **Comparative performance analysis**
- **Best model recommendations**
- **Business impact assessment**
- **Deployment guidelines**
- **Monitoring recommendations**

## 🔧 Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd DS2
```

2. **Install dependencies**:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter scipy
```

3. **Run Jupyter notebook**:
```bash
jupyter notebook Gershons-Avital-DS2-Jun-2025.ipynb
```

## 📚 Project Highlights

### Technical Excellence:
- **Modular Design**: Separate modules for EDA, ML, and evaluation
- **Comprehensive Testing**: Advanced goodness-of-fit analysis
- **Production Ready**: Error analysis and monitoring recommendations
- **Scalable Architecture**: Easy to extend with new models or features

### Business Value:
- **Risk Assessment**: Comprehensive loan default prediction
- **Decision Support**: Clear recommendations for loan approval
- **Performance Monitoring**: Tools for ongoing model evaluation
- **Cost Optimization**: Automated screening reduces manual review costs

## 👤 Author

**Gershon Avital**  
Data Science Project - June 2025

## 📄 License

This project is for educational and research purposes.

---

*For detailed analysis and results, please refer to the main Jupyter notebook: `Gershons-Avital-DS2-Jun-2025.ipynb`* 
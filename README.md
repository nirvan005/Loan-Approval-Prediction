# Loan Approval Prediction using Logistic Regression

A machine learning project that predicts loan approval status using Logistic Regression with comprehensive feature engineering and data transformation techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Key Insights](#key-insights)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Feature Engineering](#feature-engineering)
- [Results](#results)
- [Conclusion](#conclusion)

## 🎯 Overview

This project implements a **Logistic Regression model** to predict whether a loan application will be approved or rejected based on various applicant features. The project demonstrates extensive exploratory data analysis (EDA), feature transformation, and model optimization techniques.

## 📊 Dataset

The dataset (`loan_data.csv`) contains information about loan applicants with the following features:

### Numerical Features:

- `person_age` - Age of the applicant
- `person_income` - Annual income
- `person_emp_exp` - Employment experience in years
- `loan_amnt` - Loan amount requested
- `loan_int_rate` - Interest rate
- `loan_percent_income` - Loan amount as a percentage of income
- `cb_person_cred_hist_length` - Credit history length
- `credit_score` - Credit score

### Categorical Features:

- `person_gender` - Gender (Male/Female)
- `person_education` - Education level (High School, Associate, Bachelor, Master, Doctorate)
- `person_home_ownership` - Home ownership status (RENT, OWN, MORTGAGE, OTHER)
- `loan_intent` - Purpose of the loan (PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION)
- `previous_loan_defaults_on_file` - Previous loan default history (Yes/No)

### Target Variable:

- `loan_status` - Loan approval status (0 = Rejected, 1 = Approved)

## 🔍 Key Insights

### 1. **Impact of Categorical Features on Loan Approval**

Through effect size analysis (maximum percentage difference in approval rates):

- **Previous Loan Defaults**: Strongest predictor - applicants with no previous defaults have significantly higher approval rates
- **Loan Intent**: Different loan purposes show varying approval rates, with some purposes being more favorable
- **Home Ownership**: Homeownership status impacts loan approval probability
- **Person Gender**: Minimal impact on loan approval (considered a weak feature)
- **Person Education**: Small effect on loan approval (considered a weak feature)

### 2. **Numerical Feature Characteristics**

- **Skewness Analysis**: Most numerical features (`person_age`, `person_income`, `person_emp_exp`, `loan_amnt`, `loan_percent_income`, `cb_person_cred_hist_length`) showed positive skewness
- **Correlation Patterns**: Correlation analysis revealed relationships between features, helping identify multicollinearity
- **Credit Score**: Maintained without transformation as its actual values are more meaningful than normalized distribution

### 3. **Data Quality**

- **No Duplicates**: Dataset contains no duplicate records
- **Data Types**: Appropriate data types for all features
- **Missing Values**: Handled appropriately (if any)

## 🛠️ Technologies Used

```python
- Python 3.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- ydata-profiling (for comprehensive EDA)
- Pickle (for model serialization)
```

## 💻 Installation

```bash
# Clone the repository
git clone <your-repo-url>

# Navigate to project directory
cd Logistic-Regression-Loan-Approval

# Install required packages
pip install numpy pandas matplotlib seaborn scikit-learn ydata-profiling
```

## 🚀 Usage

### 1. Run the Notebook

Open and execute `Loan Approval.ipynb` in Jupyter Notebook or VS Code

### 2. Using the Saved Model

```python
import pickle
import pandas as pd

# Load the trained model
model = pickle.load(open('loan_approval_predict.pkl', 'rb'))

# Create sample data
sample = pd.DataFrame([{
    'person_age': 22,
    'person_gender': 'female',
    'person_education': 'Master',
    'person_income': 71948,
    'person_emp_exp': 0,
    'person_home_ownership': 'RENT',
    'loan_amnt': 35000,
    'loan_intent': 'PERSONAL',
    'loan_int_rate': 16.02,
    'loan_percent_income': 0.49,
    'cb_person_cred_hist_length': 3,
    'credit_score': 561,
    'previous_loan_defaults_on_file': 'No'
}])

# Make prediction
prediction = model.predict(sample)
print(f"Loan Status: {'Approved' if prediction[0] == 1 else 'Rejected'}")
```

## 📈 Model Performance

### Baseline Model (No Transformations)

- **Accuracy**: ~93% (10-fold cross-validation)
- **Precision**: High precision in predicting loan approvals
- **F1 Score**: Balanced performance across classes

### Optimized Model (With Transformations)

- **Accuracy**: ~93% (similar to baseline)
- **Precision**: Maintained high precision
- **F1 Score**: Consistent performance

### Performance Metrics Comparison

| Model Configuration                     | Accuracy | Precision | F1 Score | Iterations |
| --------------------------------------- | -------- | --------- | -------- | ---------- |
| No transformations                      | ~93%     | High      | ~0.92    | **100**    |
| All transformations                     | ~93%     | High      | ~0.92    | **376**    |
| Without age transform                   | ~93%     | High      | ~0.92    | **435**    |
| Without income transform                | ~93%     | High      | ~0.92    | **1000\*** |
| Without employment experience transform | ~93%     | High      | ~0.92    | **468**    |
| Without loan amount transform           | ~93%     | High      | ~0.92    | **1000\*** |
| Without loan percent income transform   | ~93%     | High      | ~0.92    | **1000\*** |
| Without credit history length transform | ~93%     | High      | ~0.92    | **467**    |

**Note**: **\*1000** indicates the model hit the maximum iteration limit without full convergence.

## 🔧 Feature Engineering

### 1. **Data Transformation Techniques**

Tested multiple transformation methods to reduce skewness:

#### PowerTransformer (Yeo-Johnson) - **Selected Method**

Applied to:

- `person_age`
- `person_income`
- `person_emp_exp`
- `loan_amnt`
- `loan_percent_income`
- `cb_person_cred_hist_length`

**Why Yeo-Johnson?**

- Best performance in reducing skewness
- Handles zero and negative values
- Creates more normally distributed features

#### Other Methods Tested:

- Square root transformation
- Log transformation (np.log1p)
- Power transformation with custom exponents

### 2. **Feature Scaling**

- **StandardScaler**: Applied after power transformation to normalize feature ranges

### 3. **Encoding Strategies**

#### One-Hot Encoding (drop='first' to avoid multicollinearity):

- `person_gender`
- `person_home_ownership`
- `loan_intent`
- `previous_loan_defaults_on_file`

#### Ordinal Encoding:

- `person_education` - Ordered categories: High School < Associate < Bachelor < Master < Doctorate

### 4. **Pipeline Architecture**

```python
Pipeline([
    ('transform_scale_encode', ColumnTransformer([
        ('ohe', OneHotEncoder(...), [...categorical_features]),
        ('oe', OrdinalEncoder(...), ['person_education']),
        ('num', Pipeline([
            ('power', PowerTransformer()),
            ('scaler', StandardScaler())
        ]), [...numerical_features])
    ])),
    ('logReg', LogisticRegression(max_iter=1000, tol=1e-3))
])
```

## 📊 Results

### Detailed Iteration Analysis

The impact of data transformations on model convergence:

| Configuration                             | Iterations | Convergence Status |
| ----------------------------------------- | ---------- | ------------------ |
| Without any transformations               | **100**    | ✅ Converged       |
| With all transformations                  | **376**    | ✅ Converged       |
| All transformations except age            | **435**    | ✅ Converged       |
| All transformations except income         | **1000**   | ⚠️ Max limit       |
| All transformations except employment exp | **468**    | ✅ Converged       |
| All transformations except loan amount    | **1000**   | ⚠️ Max limit       |
| All transformations except loan % income  | **1000**   | ⚠️ Max limit       |
| All transformations except credit history | **467**    | ✅ Converged       |

### Key Findings:

1. **Transformation Impact on Training Efficiency**
   - ⚠️ **Critical Features**: `person_income`, `loan_amnt`, and `loan_percent_income` are essential - without their transformations, the model fails to converge (hits 1000 iteration limit)
   - ✅ **Convergence Behavior**: While baseline model converges quickly (100 iterations), comprehensive transformations require more iterations (376) but ensure stable convergence across all feature combinations
   - ⚖️ **Accuracy**: Performance metrics remained consistent (~93%) regardless of iteration count
   - 🎯 **Best Practice**: Apply complete transformations to ensure robust convergence, especially for income-related features

2. **Feature Importance Analysis**
   - Previous loan defaults: Most significant predictor
   - Loan intent and home ownership: Strong indicators
   - Gender and education: Weak predictors

3. **Model Robustness**
   - Consistent performance across 10-fold cross-validation
   - No significant overfitting observed
   - Reliable predictions on test set

## 🎓 Conclusion

### Main Takeaways:

1. **Data Transformation Benefits**
   - Ensures reliable convergence across all feature configurations
   - Critical transformations (income, loan amount, loan % income) prevent convergence failures
   - While baseline converges fast (100 iterations), comprehensive transformations (376 iterations) provide stability
   - Doesn't improve accuracy but ensures consistent computational behavior
   - Essential for production deployment where reliability matters more than raw speed

2. **Feature Selection Insights**
   - Not all features contribute equally to predictions
   - Previous loan history is the strongest predictor
   - Gender and education have minimal impact on approval decisions

3. **Model Reliability**
   - Logistic Regression achieves ~93% accuracy with proper preprocessing
   - Model is stable across different feature configurations
   - Suitable for loan approval prediction tasks

4. **Best Practices Implemented**
   - Comprehensive EDA before modeling
   - Multiple transformation techniques tested
   - Pipeline-based approach for reproducibility
   - Cross-validation for robust performance estimation

### Future Improvements:

- Try ensemble methods (Random Forest, XGBoost) for comparison
- Implement SMOTE or other techniques if class imbalance exists
- Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- Develop a web application for real-time predictions
- Analyze feature importance using coefficients

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**Nirvan**

## 📚 Data Source

**Dataset**: [Loan Approval Classification Dataset](https://www.kaggle.com/)  
**Description**: Synthetic Data for binary classification on Loan Approval  
**Author**: Ta-wei Lo  
**Platform**: Kaggle

---

⭐ **If you found this project helpful, please give it a star!** ⭐

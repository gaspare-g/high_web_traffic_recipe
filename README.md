# Recipe High Traffic Classification

## 📊 Project Overview

This is a **DataCamp Data Scientist Professional Practical Exam** submission that develops a machine learning classification model to predict which recipes will drive high traffic to a recipe website.

**Certification**: Data Scientist Professional Certification | DataCamp

The project demonstrates a complete end-to-end data science workflow: data validation and cleaning, exploratory analysis, feature engineering, model development, and business-focused evaluation metrics.

---

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Installation & Virtual Environment Setup](#installation--virtual-environment-setup)
- [Data Validation](#data-validation)
- [Exploratory Analysis](#exploratory-analysis)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Business Metrics](#business-metrics)
- [Recommendations](#recommendations)

---

## 🗂️ Project Structure

```
recipe_high_traffic_classification/
├── validation/
│   ├── __init__.py
│   └── data_validator.py          # Data cleaning and validation logic
├── visualization/
│   ├── __init__.py
│   └── data_visualizer.py         # Visualization and plotting utilities
├── model_development/
│   ├── __init__.py
│   └── model_classes.py           # Model classes (Logistic Regression, Random Forest)
├── model_evaluation/
│   ├── __init__.py
│   └── model_evaluator.py         # Model evaluation metrics and comparison
├── notebook.ipynb                  # Main analysis and model training notebook
├── recipe_site_traffic_2212.csv    # Input dataset
├── pyproject.toml                  # Project configuration with UV
├── README.md                       # This file
```

### Module Architecture

- **validation**: `DataValidator` class handles data cleaning, imputation, and outlier detection
- **visualization**: `DataVisualizer` and `ModelVisualizer` classes provide comprehensive plotting utilities
- **model_development**: `LogisticRegressionModel` and `RandomForestModel` wrapper classes
- **model_evaluation**: `ModelEvaluator` class for metrics calculation and model comparison

---

## 🔧 Installation & Virtual Environment Setup

### Windows Setup

1. **Navigate to the project directory**:
```bash
cd recipe_high_traffic_classification
```

2. **Create and activate the virtual environment**:
```bash
uv venv
.venv\Scripts\activate
```

3. **Install all dependencies**:
```bash
uv sync
```

This command will install all dependencies specified in `pyproject.toml`.

4. **Run the Jupyter Notebook**:
```bash
jupyter notebook notebook.ipynb
```

---

## 📊 Data Validation

### Initial Observations

- **947 entries** with multiple columns for recipe attributes
- **373 missing values** in `high_traffic` column (only records with "High" traffic were coded)
- **52 missing values** across nutritional columns (calories, carbohydrate, sugar, protein)
- **Category anomaly**: "Chicken Breast" found instead of just "Chicken"
- **Servings data type**: Mixed text/numeric values ("4 as a snack", "6 as a snack")

### Cleaning & Validation Steps

1. **Category Standardization**: Recoded "Chicken Breast" → "Chicken"
2. **Servings Cleaning**: Extracted numeric values from text entries
3. **Target Recoding**: Converted `high_traffic` to boolean (True/False)
4. **Missing Value Imputation**: Used category-wise mean imputation for numerical features

### Validation Results

```
✅ All 947 recipe IDs are unique
✅ All 10 expected recipe categories present
✅ No missing values after imputation
✅ Imputed values preserve original distributions
```

---

## 🔍 Exploratory Analysis

### Key Findings

#### Univariate Distributions
- **Numerical Features**: All show right-skewed distributions (calories, carbohydrate, sugar, protein)
- **Target Variable**: 60.61% of recipes are classified as high-traffic (baseline accuracy)
- **Most Popular Categories**: Chicken, Potato, Vegetable, Pork
- **Least Popular**: Beverages, Breakfast, Meat

#### Bivariate Analysis
- **Strong Correlation**: `category` shows very high chi-squared correlation with `high_traffic` (p < 0.001)
- **Modest Numerical Correlation**: Nutritional features show modest non-linear patterns with target
- **No Correlation**: `servings` shows no significant association with either category or target

#### Non-Linear Patterns
Binned means analysis reveals non-linear relationships in:
- Calories vs High Traffic
- Carbohydrate vs High Traffic  
- Sugar vs High Traffic

#### Outlier Analysis
- 52 outliers detected using z-score method within categories
- Outliers show **NO significant correlation** with target variable
- Safe to include outliers without concerns of bias

---

## 🤖 Model Development

### Problem Statement

**Classification Task**: Predict whether recipes will drive high website traffic (binary classification)

### Model Selection Strategy

#### Model 1: Logistic Regression (Baseline)
- **Why**: Simple, interpretable, establishes baseline
- **Features**: Category only (one-hot encoded)
- **Rationale**: Avoids multicollinearity issues; numerical features highly correlated with category
- **Hyperparameters**: `C=109.85`, `penalty='l2'`, `solver='lbfgs'`

#### Model 2: Random Forest Classifier (Comparison)
- **Why**: Captures non-linear relationships, robust to multicollinearity
- **Features**: All features including log-transformed numerical values
- **Rationale**: Tests if non-linear patterns improve prediction
- **Hyperparameters**: `n_estimators=10`, `max_features='log2'`, `min_samples_split=15`

### Feature Engineering

```python
# Log transformation of numerical features
- Created: calories_log, sugar_log, carbohydrate_log

# Categorical encoding
- One-hot encoded category variable (drop_first=True)
```

### Train-Test Split
- **Training Set**: 75% (711 samples)
- **Test Set**: 25% (236 samples)
- **Random Seed**: 42 (reproducibility)

---

## 📈 Model Evaluation

### Evaluation Metrics

**Test Set Performance Comparison**:

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|-----------------|
| Accuracy | 0.8390 | 0.8220 |
| Precision | 0.8170 | 0.7917 |
| Recall | 0.8121 | 0.7805 |
| ROC AUC | 0.8805 | 0.8563 |

### Key Findings

✅ **Logistic Regression is the Recommended Model**
- Higher precision (0.817 vs 0.792)
- Higher recall (0.812 vs 0.781)
- No overfitting (test ≈ train metrics)
- Simpler, more interpretable

⚠️ **Random Forest Results**
- Slightly lower performance across all metrics
- Moderate overfitting detected (train metrics > test metrics)
- Non-linear patterns not substantial enough to warrant complexity

### Confusion Matrix Analysis

**Logistic Regression Test Set**:
- True Positives: 121 (correctly identified high-traffic recipes)
- True Negatives: 97 (correctly identified low-traffic recipes)
- False Positives: 27 (falsely predicted as high-traffic)
- False Negatives: 28 (missed high-traffic recipes)

---

## 💼 Business Metrics

### Business Context

**Prior Baseline**: Manual recipe selection achieved **60.61% precision** in identifying high-traffic recipes

### Model Performance vs Baseline

```
Model Precision:      81.7%
Baseline Precision:   60.61%
Improvement:          +21.09 percentage points
```

### Estimated Web Traffic Impact

**Calculation**:
```
Expected Traffic Increase = 40% × (Model Precision - Baseline Precision)
                         = 40% × (0.817 - 0.606)
                         = 40% × 0.211
                         = ~8.44% increase in web traffic
```

### Key Performance Indicators (KPIs) to Monitor

1. **Model Metrics**
   - Precision: Maintain ≥ 80% (proportion of true high-traffic predictions)
   - Recall: Maintain ≥ 80% (coverage of actual high-traffic recipes)

2. **Business Metrics**
   - Monthly web traffic increase (target: +8.44%)
   - Model coefficient changes (recipe category preferences may shift)

3. **Category Insights**
   - **Strong Predictors of High Traffic**: Pork, Potato, Vegetable
   - **Weak Predictors**: Beverages, Breakfast, Chicken

---

## 🎯 Recommendations

### Immediate Actions

1. **Deploy Logistic Regression Model**
   - Implement in production for homepage recipe selection
   - Use category-encoded feature set
   - Monitor model predictions weekly

2. **Monitor Performance KPIs**
   - Track precision and recall monthly
   - Log actual web traffic metrics
   - Create alert if performance drops below 75%

3. **Conduct A/B Testing**
   - Test model-based selection vs human selection
   - Measure actual traffic impact
   - Validate business assumptions

### Strategic Actions

4. **Analyze Category Trends**
   - Focus content development on Pork, Potato, Vegetable recipes
   - Investigate why Beverages, Breakfast, Chicken underperform
   - Consider user demographics and seasonal preferences

5. **Continuous Improvement**
   - Collect new labeled data quarterly
   - Retrain model if precision/recall degrades
   - Evaluate new features (cooking time, difficulty, etc.)

6. **Model Governance**
   - Document all model changes in versioning system
   - Establish retraining schedule (quarterly recommended)
   - Create data quality checks for incoming recipes

---

## 📁 Dataset Information

**File**: `recipe_site_traffic_2212.csv`

### Columns

- **recipe** (int): Unique recipe identifier (1-947)
- **calories** (float): Caloric content per serving
- **carbohydrate** (float): Carbohydrate grams per serving
- **sugar** (float): Sugar grams per serving
- **protein** (float): Protein grams per serving
- **category** (string): Recipe category (10 types)
- **servings** (int): Number of servings
- **high_traffic** (boolean): Whether recipe drove high website traffic

---

## 🛠️ Technical Stack

- **Python**: 3.9+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Jupyter**: For interactive notebook analysis
- **Virtual Environment**: UV

---

## 📝 Code Quality

### Style & Standards
- **Black Formatter**: Consistent code style
- **Pylint**: Code quality checks
- **Docstrings**: Comprehensive module and function documentation
- **Type Hints**: Function argument documentation

### Module-Based Architecture
All code is organized into reusable classes and modules:
- Single Responsibility Principle
- DRY (Don't Repeat Yourself)
- Easy to test and maintain

---

## 🔗 References

- **DataCamp Professional Data Scientist Certification**: [DataCamp](https://www.datacamp.com/certification/data-scientist)
- **UV Package Manager**: [UV Documentation](https://github.com/astral-sh/uv)
- **Scikit-Learn**: [Official Documentation](https://scikit-learn.org/)
- **Jupyter**: [Official Website](https://jupyter.org/)

---

## 📄 License

This project is part of a DataCamp certification submission.

---

## 👤 Author

**Project**: Recipe High Traffic Classification  
**Submission**: DataCamp Data Scientist Professional Practical Exam  
**Result**: Successful


---

## 📞 Support

For questions or issues:
1. Check the detailed notebook analysis in `notebook.ipynb`
2. Review module docstrings in the respective package folders
3. Consult the original analysis markdown sections in the notebook

---

**Project Status**: ✅ Completed  
**Model Status**: ✅ Production Ready  
**Certification**: ✅ DataCamp Data Scientist Professional

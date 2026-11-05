# 🥗 Medical Condition-Based Diet Recommendation Using Machine Learning

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/pandas-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/numpy-013243.svg?logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent clinical decision support prototype that combines Supervised Machine Learning with the **USDA National Nutrient Database** to assess patient health metrics, classify optimal dietary strategies (**Balanced**, **Low_Sodium**, **Low_Carb**), and generate tailored, nutrient-optimized food recommendations.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Project Objectives](#-project-objectives)
- [Dataset Architecture](#-dataset-architecture)
  - [1. Patient Health & Clinical Dataset](#1-patient-health--clinical-dataset)
  - [2. USDA National Nutrient Database](#2-usda-national-nutrient-database)
- [Machine Learning Workflow](#-machine-learning-workflow)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Algorithms Implemented](#algorithms-implemented)
  - [Model Evaluation & Benchmarking](#model-evaluation--benchmarking)
- [USDA Food Recommendation Engine](#-usda-food-recommendation-engine)
- [Interactive Clinical Prediction Session](#-interactive-clinical-prediction-session)
- [Project Structure](#-project-structure)
- [Generated Artifacts](#-generated-artifacts)
- [Installation & Setup](#-installation--setup)
- [Execution Guide](#-execution-guide)
- [Clinical Disclaimer](#-clinical-disclaimer)
- [Author & Acknowledgments](#-author--acknowledgments)

---

## 📖 Overview

Dietary management is a cornerstone in the treatment and prevention of chronic lifestyle conditions such as diabetes, hypertension, and obesity. However, connecting broad medical guidelines to specific, actionable food choices is challenging.

This project bridges that gap by implementing a dual-stage system:
1. **Machine Learning Classifier**: Evaluates a patient's biometrics, disease type, severity, lab values (glucose, cholesterol, blood pressure), and lifestyle factors to predict their target diet category (`Balanced`, `Low_Sodium`, or `Low_Carb`).
2. **Nutritional Recommender**: Queries the extensive USDA food composition database (7,000+ items) to shortlist and rank real-world foods matching the macronutrient and micronutrient targets dictated by the predicted diet regimen.

---

## 🎯 Project Objectives

1. **Patient Risk & Diet Stratification**: Classify patient profiles into clinically aligned diet categories using multiclass classification.
2. **Leakage-Free Preprocessing**: Construct robust numerical and categorical Scikit-Learn transformation pipelines with median/modal imputation, z-score scaling, and one-hot encoding.
3. **Multi-Algorithm Benchmarking**: Evaluate and compare Decision Trees, Random Forests, Logistic Regression, and Support Vector Machines (SVM with RBF kernel).
4. **USDA Nutrient Integration**: Dynamically filter and rank foods based on nutritional criteria (e.g., minimizing sodium while maximizing protein for hypertensive profiles).
5. **Model Serialization**: Export model comparison tables and serialize the champion classifier using `joblib` for immediate reuse.
6. **Interactive Consultation Interface**: Live CLI interface allowing users or clinicians to input custom patient metrics and obtain instant diet predictions and food shortlists.

---

## 📊 Dataset Architecture

### 1. Patient Health & Clinical Dataset
- **File**: `diet_recommendations_dataset.csv`
- **Observations**: 1,000 patient records across 20 attributes.
- **Target Variable**: `Diet_Recommendation`
  - `Balanced`: 426 samples (42.6%)
  - `Low_Sodium`: 316 samples (31.6%)
  - `Low_Carb`: 258 samples (25.8%)

#### Patient Feature Inventory
| Category | Attributes |
| :--- | :--- |
| **Demographics & Physical** | `Age`, `Gender`, `Weight_kg`, `Height_cm`, `BMI` |
| **Medical & Clinical** | `Disease_Type` *(Diabetes, Hypertension, Obesity)*, `Severity` *(Mild, Moderate, Severe)*, `Cholesterol_mg/dL`, `Blood_Pressure_mmHg`, `Glucose_mg/dL` |
| **Lifestyle & Habits** | `Physical_Activity_Level`, `Daily_Caloric_Intake`, `Weekly_Exercise_Hours`, `Adherence_to_Diet_Plan`, `Dietary_Nutrient_Imbalance_Score` |
| **Dietary Preferences** | `Dietary_Restrictions`, `Allergies`, `Preferred_Cuisine` |
| **Identifiers (Excluded)** | `Patient_ID` *(Removed prior to training)* |

### 2. USDA National Nutrient Database
- **File**: `USDA_cleaned.csv` (and original `USDA.csv`)
- **Records**: 7,057 comprehensive food items.
- **Nutritional Metrics (per 100g)**: `Calories`, `Protein`, `TotalFat`, `Carbohydrate`, `Sodium`, `SaturatedFat`, `Cholesterol`, `Sugar`, `Fiber`.

---

## ⚙️ Machine Learning Workflow

```
[Patient Clinical Dataset: 1,000 Records]
                     │
                     ▼
[Exclusion of Non-Predictive ID: Patient_ID]
                     │
                     ▼
[Stratified 80/20 Train-Test Split]
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
[Numerical Pipeline]     [Categorical Pipeline]
 - SimpleImputer(median)   - SimpleImputer("Unknown")
 - StandardScaler          - OneHotEncoder(ignore)
        │                         │
        └────────────┬────────────┘
                     ▼
            [ColumnTransformer]
                     │
     ┌───────────────┼───────────────┬───────────────┐
     ▼               ▼               ▼               ▼
[Decision Tree] [Random Forest] [Logistic Reg]     [SVM - RBF]
 (depth=5)     (300 trees)     (iter=3000)      (C=1.0, RBF)
     │               │               │               │
     └───────────────┼───────────────┴───────────────┘
                     ▼
        [Performance Evaluation]
   - Accuracy, Precision, Recall, F1-Score
   - 2x2 Confusion Matrix Displays
                     │
                     ▼
     [Model Selection & Serialization]
   - best_diet_classifier.joblib
   - model_comparison.csv
                     │
                     ▼
[USDA Nutritional Recommender & Interactive Patient Assessment]
```

### Preprocessing Pipeline
- **Numerical Pipeline**: Missing values are imputed with median values, followed by standard z-score scaling (`StandardScaler`).
- **Categorical Pipeline**: Missing strings are imputed with `"Unknown"`, followed by one-hot encoding (`OneHotEncoder(handle_unknown="ignore")`).
- **Encapsulation**: Bundled inside `ColumnTransformer` and integrated directly into Scikit-Learn `Pipeline` instances to prevent train/test data leakage.

### Algorithms Implemented
1. **Decision Tree Classifier**:
   - `max_depth=5`, `class_weight="balanced"`, `random_state=42`
2. **Random Forest Classifier**:
   - `n_estimators=300`, `min_samples_leaf=2`, `class_weight="balanced"`, `random_state=42`
3. **Logistic Regression**:
   - Multi-class multinomial log-odds modeling with `max_iter=3000`, `class_weight="balanced"`, `random_state=42`
4. **Support Vector Machine (SVM / SVC)**:
   - Non-linear radial basis function kernel: `kernel="rbf"`, `C=1.0`, `class_weight="balanced"`

### Model Evaluation & Benchmarking
Models are assessed using:
- **Accuracy**
- **Weighted Precision**
- **Weighted Recall**
- **Weighted F1-Score** (Primary ranking metric)
- **Confusion Matrix Grid ($2 \times 2$)**: Visual breakdown of true vs. predicted classifications across all four models.

---

## 🥗 USDA Food Recommendation Engine

Once the classifier determines the optimal diet plan for a patient, the recommendation engine queries `USDA_cleaned.csv` and applies nutritional sorting rules:

- **Low_Carb**:
  - Filters out records with missing carbohydrate data.
  - Sorts primarily by lowest `Carbohydrate`, then by highest `Protein`.
- **Low_Sodium**:
  - Filters out records with missing sodium data.
  - Sorts primarily by lowest `Sodium`, then by highest `Protein`.
- **Balanced**:
  - Filters out calorie-dense outliers ($\le 75\text{th}$ percentile of `Calories`).
  - Balances nutrient density by sorting by highest `Protein` followed by controlled `Carbohydrate`.

The engine returns the **Top 15** foods with their full nutritional breakdown per 100g.

---

## 💬 Interactive Clinical Prediction Session

The script provides a live interactive session where a user or healthcare professional can input custom parameters:

```plaintext
============================================================
PERSONAL DIET CATEGORY PREDICTION
============================================================

Enter the person's details below.
Press Enter to use the example default.

Disease types in the dataset: Diabetes, Hypertension, Obesity

Age (years) [default: 45]: 52
Gender (as in dataset) [default: Female]: Male
Weight in kg [default: 78.0]: 84.5
Height in cm [default: 165.0]: 172.0
BMI [default: 28.7]: 28.6
Disease type [default: Diabetes]: Hypertension
Severity [default: Moderate]: Severe
Physical activity level [default: Moderate]: Low
Daily caloric intake [default: 2100.0]: 2300
Cholesterol (mg/dL) [default: 190.0]: 210
Systolic blood pressure (mmHg) [default: 128.0]: 155
Glucose (mg/dL) [default: 145.0]: 110
Dietary restrictions (or None) [default: None]: None
Allergies (or Unknown) [default: Unknown]: Peanut
Preferred cuisine [default: Indian]: Mediterranean
Weekly exercise hours [default: 3.0]: 1.5
Diet-plan adherence (0-100) [default: 60.0]: 50
Nutrient imbalance score [default: 2.0]: 3.0

============================================================
PREDICTION RESULT
============================================================
Disease type entered: Hypertension
Predicted diet category: Low_Sodium
BMI calculated from height and weight: 28.56

Suggested foods from the USDA dataset:
(Nutrient values are generally reported per 100 g):
[Top 15 Low Sodium, High Protein USDA Foods Shortlisted]
```

---

## 📁 Project Structure

```plaintext
Diet_Recommendation_System/
│
├── diet_recommendation_ml.py           # Complete ML pipeline, evaluation, and recommender
├── diet_recommendations_dataset.csv    # Patient health and clinical dataset (1,000 records)
├── USDA_cleaned.csv                    # Cleaned USDA food nutritional database (7,057 records)
├── USDA.csv                            # Original USDA food composition database
├── requirements.txt                    # Project library dependencies
├── .gitignore                          # Git exclusion rules (cache, venv, IDE artifacts)
├── README.md                           # Comprehensive documentation
│
└── diet_project_outputs/               # Generated reports and serialized model
    ├── best_diet_classifier.joblib     # Persisted top-performing Scikit-Learn pipeline
    ├── model_comparison.csv            # Benchmarking metrics table for all 4 algorithms
    ├── food_suggestions_balanced.csv   # Top 15 recommended balanced foods
    ├── food_suggestions_low_carb.csv   # Top 15 recommended low-carbohydrate foods
    └── food_suggestions_low_sodium.csv # Top 15 recommended low-sodium foods
```

---

## 📦 Generated Artifacts

Executing the script automatically produces the following in `diet_project_outputs/`:
- **`best_diet_classifier.joblib`**: Serialized pipeline ready for deployment in web apps, APIs, or clinical dashboards.
- **`model_comparison.csv`**: Comparison metrics across Decision Tree, Random Forest, Logistic Regression, and SVM.
- **`food_suggestions_*.csv`**: Ready-to-use CSV food shortlists for each supported dietary regimen.

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/HemanthSai06/diet_recommendation.git
cd diet_recommendation
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# On macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Execution Guide

Run the complete pipeline from the command line:

```bash
python diet_recommendation_ml.py
```

*Note: Execution will train all 4 classifiers, display confusion matrices, output comparison metrics, save the model and CSV recommendations, and prompt for new patient input for live inference.*

---

## ⚠️ Clinical Disclaimer

This software is an educational machine learning and data science prototype. The predictions and food suggestions are derived from statistical patterns in the datasets and simple nutrient-based heuristics. They do not constitute certified medical, diagnostic, or dietary advice. Always consult a qualified physician or registered dietitian before modifying nutrition or medical treatments.

---

## 👤 Author

- **Hemanth Sai**  
  GitHub: [@HemanthSai06](https://github.com/HemanthSai06)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.


"""
MEDICAL CONDITION-BASED DIET RECOMMENDATION USING MACHINE LEARNING

Algorithms:
1. Decision Tree
2. Random Forest
3. Logistic Regression
4. Support Vector Machine (SVM)

Required files:
- diet_recommendations_dataset.csv
- USDA_cleaned.csv

Plots are displayed in Spyder and are NOT saved as image files.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# ==========================================================
# 1. FILE PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

PATIENT_FILE = BASE_DIR / "/Users/phemanthsai/Desktop/Ml_Pro/Diet_Recommendation_System/diet_recommendations_dataset.csv"
USDA_FILE = BASE_DIR / "/Users/phemanthsai/Desktop/Ml_Pro/Diet_Recommendation_System/USDA_cleaned.csv"

OUTPUT_DIR = BASE_DIR / "diet_project_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================================
# 2. LOAD DATASETS
# ==========================================================

def load_data():

    if not PATIENT_FILE.exists():
        raise FileNotFoundError(
            f"Patient dataset not found:\n{PATIENT_FILE}"
        )

    if not USDA_FILE.exists():
        raise FileNotFoundError(
            f"USDA dataset not found:\n{USDA_FILE}"
        )

    patients = pd.read_csv(PATIENT_FILE)
    foods = pd.read_csv(USDA_FILE)

    if "Diet_Recommendation" not in patients.columns:
        raise ValueError(
            "The patient dataset must contain "
            "'Diet_Recommendation'."
        )

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print("\nPatient dataset shape:", patients.shape)
    print("USDA dataset shape:", foods.shape)

    print("\nDiet recommendation distribution:")
    print(patients["Diet_Recommendation"].value_counts())

    print("\nMissing values in patient dataset:")
    print(patients.isnull().sum()[patients.isnull().sum() > 0])

    return patients, foods


# ==========================================================
# 3. TRAIN AND EVALUATE FOUR ML ALGORITHMS
# ==========================================================

def train_and_evaluate(patients):

    target = "Diet_Recommendation"

    data = patients.dropna(subset=[target]).copy()

    # Patient_ID is an identifier, not a useful health feature.
    X = data.drop(
        columns=[target, "Patient_ID"],
        errors="ignore"
    )

    y = data[target].astype(str)

    # This dataset has a strong, direct relationship between
    # disease type and diet labels. The accuracy measures
    # dataset prediction, not clinical validity.

    if "Disease_Type" in data.columns:

        print("\nDisease type versus diet label:")
        print(
            pd.crosstab(
                data["Disease_Type"].fillna("MISSING"),
                y
            )
        )

        print(
            "\nNOTE: Disease_Type strongly determines the labels "
            "in this dataset. High accuracy does not establish "
            "medical reliability."
        )

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    # Preprocessing for numerical features.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Preprocessing for categorical features.
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown"
                )
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            ),
        ],
        remainder="drop"
    )

    # Four classification algorithms.
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=42,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ),

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            class_weight="balanced"
        ),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    results = []
    trained_models = {}

    labels = sorted(y.unique())

    print("\n" + "=" * 60)
    print("MODEL TRAINING AND EVALUATION")
    print("=" * 60)

    # Create four confusion-matrix plots.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for index, (name, classifier) in enumerate(models.items()):

        model = Pipeline(
            steps=[
                ("preprocessing", preprocessing),
                ("classifier", classifier),
            ]
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
        )

        results.append({
            "Algorithm": name,
            "Accuracy": accuracy,
            "Weighted Precision": precision,
            "Weighted Recall": recall,
            "Weighted F1-score": f1,
        })

        trained_models[name] = model

        print("\n" + "-" * 50)
        print("Algorithm:", name)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y_test,
                predictions,
                zero_division=0
            )
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=labels
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=labels
        )

        display.plot(
            ax=axes[index],
            cmap="Blues",
            values_format="d",
            colorbar=False
        )

        axes[index].set_title(name)

    fig.suptitle(
        "Confusion Matrices for Diet Classification",
        fontsize=14
    )

    fig.tight_layout()

    # Show plots in Spyder. Do NOT save plots to disk.
    plt.show()

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by=["Weighted F1-score", "Accuracy"],
        ascending=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # Save the model comparison table (not a plot).
    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False
    )

    best_name = results_df.iloc[0]["Algorithm"]
    best_model = trained_models[best_name]

    # Save the trained model for future use.
    joblib.dump(
        best_model,
        OUTPUT_DIR / "best_diet_classifier.joblib"
    )

    print("\nSelected model:", best_name)
    print("Model and comparison CSV saved in:", OUTPUT_DIR)

    return best_model, results_df


# ==========================================================
# 4. FOOD RECOMMENDATION USING USDA DATA
# ==========================================================

def recommend_foods(foods, diet_category, top_n=15):

    required_columns = {
        "Description",
        "Calories",
        "Protein",
        "Carbohydrate",
        "Sodium"
    }

    missing_columns = required_columns - set(foods.columns)

    if missing_columns:
        raise ValueError(
            "Missing required USDA columns: "
            + ", ".join(sorted(missing_columns))
        )

    result = foods.copy()

    numeric_columns = [
        "Calories",
        "Protein",
        "Carbohydrate",
        "Sodium",
        "Sugar",
        "TotalFat",
        "Fiber"
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce"
            )

    category = str(diet_category).strip().lower()

    if category == "low_carb":

        # Lower carbohydrate foods first.
        result = result.dropna(subset=["Carbohydrate"])

        result = result.sort_values(
            by=["Carbohydrate", "Protein"],
            ascending=[True, False]
        )

    elif category == "low_sodium":

        # Lower sodium foods first.
        result = result.dropna(subset=["Sodium"])

        result = result.sort_values(
            by=["Sodium", "Protein"],
            ascending=[True, False]
        )

    else:

        # Basic food shortlist for a balanced category.
        result = result.dropna(
            subset=["Calories", "Protein", "Carbohydrate"]
        )

        calorie_limit = result["Calories"].quantile(0.75)

        result = result[
            result["Calories"] <= calorie_limit
        ]

        result = result.sort_values(
            by=["Protein", "Carbohydrate"],
            ascending=[False, True]
        )

    display_columns = [
        column for column in [
            "ID",
            "Description",
            "Calories",
            "Protein",
            "TotalFat",
            "Carbohydrate",
            "Sodium",
            "Sugar",
            "Fiber"
        ]
        if column in result.columns
    ]

    return result[display_columns].head(top_n).copy()


# ==========================================================
# 5. COLLECT DETAILS FROM A NEW USER
# ==========================================================

def ask_value(label, default, cast=str):

    while True:

        raw = input(
            f"{label} [default: {default}]: "
        ).strip()

        if not raw:
            return default

        try:
            return cast(raw)

        except ValueError:
            print(
                f"Invalid value. Please enter a valid "
                f"{cast.__name__} value."
            )


def predict_new_patient(best_model, foods):

    print("\n" + "=" * 60)
    print("PERSONAL DIET CATEGORY PREDICTION")
    print("=" * 60)

    print(
        "\nEnter the person's details below."
        "\nPress Enter to use the example default."
    )

    print(
        "\nDisease types in the dataset: "
        "Diabetes, Hypertension, Obesity"
    )

    # Collect the feature values expected by the model.
    patient = {
        "Age": ask_value("Age (years)", 45, int),

        "Gender": ask_value(
            "Gender (as in dataset)",
            "Female"
        ),

        "Weight_kg": ask_value(
            "Weight in kg",
            78.0,
            float
        ),

        "Height_cm": ask_value(
            "Height in cm",
            165.0,
            float
        ),

        "BMI": ask_value(
            "BMI",
            28.7,
            float
        ),

        "Disease_Type": ask_value(
            "Disease type",
            "Diabetes"
        ),

        "Severity": ask_value(
            "Severity",
            "Moderate"
        ),

        "Physical_Activity_Level": ask_value(
            "Physical activity level",
            "Moderate"
        ),

        "Daily_Caloric_Intake": ask_value(
            "Daily caloric intake",
            2100.0,
            float
        ),

        "Cholesterol_mg/dL": ask_value(
            "Cholesterol (mg/dL)",
            190.0,
            float
        ),

        "Blood_Pressure_mmHg": ask_value(
            "Systolic blood pressure (mmHg)",
            128.0,
            float
        ),

        "Glucose_mg/dL": ask_value(
            "Glucose (mg/dL)",
            145.0,
            float
        ),

        "Dietary_Restrictions": ask_value(
            "Dietary restrictions (or None)",
            "None"
        ),

        "Allergies": ask_value(
            "Allergies (or Unknown)",
            "Unknown"
        ),

        "Preferred_Cuisine": ask_value(
            "Preferred cuisine",
            "Indian"
        ),

        "Weekly_Exercise_Hours": ask_value(
            "Weekly exercise hours",
            3.0,
            float
        ),

        "Adherence_to_Diet_Plan": ask_value(
            "Diet-plan adherence (0-100)",
            60.0,
            float
        ),

        "Dietary_Nutrient_Imbalance_Score": ask_value(
            "Nutrient imbalance score",
            2.0,
            float
        ),
    }

    # Use the same feature order/names as the training data.
    patient_df = pd.DataFrame([patient])

    # Predict the diet category.
    prediction = str(best_model.predict(patient_df)[0])

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print("Disease type entered:", patient["Disease_Type"])
    print("Predicted diet category:", prediction)

    # Calculate BMI for reference from weight and height.
    height_m = patient["Height_cm"] / 100

    if height_m > 0:
        calculated_bmi = patient["Weight_kg"] / (height_m ** 2)
        print(f"BMI calculated from height and weight: {calculated_bmi:.2f}")

    print("\nSuggested foods from the USDA dataset")
    print("(Nutrient values are generally reported per 100 g):")

    try:
        suggestions = recommend_foods(
            foods,
            prediction,
            top_n=15
        )

        if suggestions.empty:
            print("No matching food records were found.")
        else:
            print(suggestions.to_string(index=False))

    except (ValueError, KeyError) as error:
        print("Unable to create food shortlist:", error)

    print("\nIMPORTANT:")
    print(
        "- This is an educational machine-learning prototype."
    )
    print(
        "- The prediction reflects patterns in the provided dataset."
    )
    print(
        "- The food shortlist is based on simple nutrient sorting."
    )
    print(
        "- Allergies and dietary restrictions are collected but "
        "are not reliably checked against the USDA data."
    )
    print(
        "- Confirm food safety, portions, medications and diet "
        "changes with a qualified healthcare professional."
    )

    return prediction


# ==========================================================
# 6. MAIN PROGRAM
# ==========================================================

def main():

    patients, foods = load_data()

    best_model, results = train_and_evaluate(patients)

    # Save example food lists as CSV files only.
    # No plot images are saved.
    for category in [
        "Low_Carb",
        "Low_Sodium",
        "Balanced"
    ]:

        try:
            food_list = recommend_foods(
                foods,
                category,
                top_n=15
            )

            filename = (
                f"food_suggestions_{category.lower()}.csv"
            )

            food_list.to_csv(
                OUTPUT_DIR / filename,
                index=False
            )

        except ValueError as error:
            print(
                f"Could not create {category} food list: {error}"
            )

    # Ask for new patient details and predict a diet category.
    predict_new_patient(best_model, foods)

    print("\n" + "=" * 60)
    print("PROGRAM COMPLETED")
    print("=" * 60)

    print("Plots were displayed and were NOT saved as images.")
    print("CSV reports and the trained model are in:")
    print(OUTPUT_DIR)

    print(
        "\nThe dataset's high accuracy is not proof of "
        "clinical reliability."
    )


if __name__ == "__main__":
    main()

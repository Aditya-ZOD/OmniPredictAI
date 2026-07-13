# Script for loading the heart disease dataset, training multiple ML models,
# evaluating them, and saving the best model artifacts and reports.

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


# Set up paths for the dataset, reports directory, and model output directory.
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "heart.csv"
MODEL_DIR = BASE_DIR / "model"
REPORTS_DIR = BASE_DIR / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# Load the dataset using pandas.
df = pd.read_csv(DATASET_PATH)

# The UCI heart disease dataset may contain missing values represented as '?' and a multi-class target.
# We clean those values and convert the target into a binary outcome for this app.
for column in df.columns:
    if df[column].dtype == "object":
        df[column] = pd.to_numeric(df[column], errors="coerce")

# Fill missing numeric values using the median for each column.
df = df.fillna(df.median(numeric_only=True))

# Convert the remaining categorical columns to numeric values so the model can train on them.
for column in ["ca", "thal"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Fill any remaining missing values introduced during conversion.
df = df.fillna(df.median(numeric_only=True))

# Convert the multi-class target to a binary label: 0 = no disease, 1 = disease present.
df["target"] = df["target"].apply(lambda value: 1 if value > 0 else 0)

# Display basic dataset information.
print("Dataset loaded successfully.")
print("\nDataset information:")
df.info()

# Check for missing values in the dataset.
print("\nMissing values per column:")
print(df.isnull().sum())

# Remove duplicate rows if they exist.
print("\nNumber of duplicate rows before removal:", df.duplicated().sum())
if df.duplicated().any():
    df = df.drop_duplicates().reset_index(drop=True)
    print("Duplicates removed.")

# Separate the feature columns from the target column.
if "target" not in df.columns:
    raise ValueError("The dataset must contain a 'target' column.")

X = df.drop(columns=["target"])
y = df["target"]

print("\nFeature columns:", list(X.columns))
print("Target column: target")

# Create a target distribution chart and save it inside the reports folder.
plt.figure(figsize=(7, 4))
df["target"].value_counts().plot(kind="bar", color=["#4c78a8", "#f58518"])
plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(REPORTS_DIR / "target_distribution.png")
plt.close()

# Create a correlation heatmap and save it inside the reports folder.
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr(numeric_only=True)
im = plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar(im, label="Correlation")
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "correlation_heatmap.png")
plt.close()

# Use a small cross-validated benchmark because the dataset is moderate in size.
# This gives a more reliable accuracy estimate than a single 80/20 split.
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=300),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
}

results = []
model_names = []
model_accuracies = []

for name, model in models.items():
    pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    mean_accuracy = scores.mean()

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Cross-validated accuracy: {mean_accuracy:.4f}")
    print(f"Fold scores: {scores}")

    model_names.append(name)
    model_accuracies.append(mean_accuracy)
    results.append((mean_accuracy, name, pipeline))

# Select the best model automatically based on the highest cross-validated accuracy.
best_accuracy, best_name, best_pipeline = max(results, key=lambda item: item[0])
best_pipeline.fit(X, y)
scaler = best_pipeline.named_steps["scaler"]
best_model = best_pipeline.named_steps["model"]
print(f"\nBest model selected: {best_name} with mean cross-validated accuracy {best_accuracy:.4f}")

# Create a model comparison graph and save it inside the reports folder.
plt.figure(figsize=(8, 4))
plt.bar(model_names, model_accuracies, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"])
plt.title("Model Comparison by Accuracy")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(REPORTS_DIR / "model_comparison.png")
plt.close()

# Create a feature importance graph using the selected model's available importance scores.
if hasattr(best_model, "feature_importances_"):
    feature_importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    feature_importances = best_model.coef_[0]
else:
    feature_importances = None

if feature_importances is not None:
    feature_names = X.columns
    importance_pairs = sorted(zip(feature_names, feature_importances), key=lambda item: item[1], reverse=True)
    names = [item[0] for item in importance_pairs]
    values = [item[1] for item in importance_pairs]

    plt.figure(figsize=(8, 4))
    plt.bar(names, values, color="#4c78a8")
    plt.title("Feature Importance")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png")
    plt.close()
else:
    print("\nFeature importance graph skipped because the selected model does not expose importance scores.")

# Save the best trained model and the scaler for later use in the Flask app.
with open(MODEL_DIR / "heart_model.pkl", "wb") as model_file:
    pickle.dump(best_model, model_file)

with open(MODEL_DIR / "scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"\nSaved best model to {MODEL_DIR / 'heart_model.pkl'}")
print(f"Saved scaler to {MODEL_DIR / 'scaler.pkl'}")

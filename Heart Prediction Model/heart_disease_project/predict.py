import pandas as pd
from datetime import datetime

from database import insert_prediction_record

# Define the feature order expected by the trained model.
FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def prepare_form_data(form_data):
    """Convert incoming form inputs into numeric values that the model can process."""
    prepared_data = {
        "patient_name": str(form_data.get("patient_name", "")).strip(),
        "age": form_data.get("age", type=float),
        "sex": form_data.get("sex", type=int),
        "cp": form_data.get("cp", type=int),
        "trestbps": form_data.get("trestbps", type=float),
        "chol": form_data.get("chol", type=float),
        "fbs": form_data.get("fbs", type=int),
        "restecg": form_data.get("restecg", type=int),
        "thalach": form_data.get("thalach", type=float),
        "exang": form_data.get("exang", type=int),
        "oldpeak": form_data.get("oldpeak", type=float),
        "slope": form_data.get("slope", type=int),
        "ca": form_data.get("ca", type=int),
        "thal": form_data.get("thal", type=int),
    }
    return prepared_data


def run_prediction(form_data, model, scaler):
    """Run the trained model on the supplied features and return the result details."""
    input_df = pd.DataFrame([form_data], columns=FEATURES)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = max(model.predict_proba(scaled_input)[0])

    if prediction == 1:
        result_text = "Positive"
        risk_level = "High Risk" if probability >= 0.75 else "Moderate Risk"
    else:
        result_text = "Negative"
        risk_level = "Low Risk" if probability >= 0.75 else "Moderate Risk"

    confidence_percentage = round(probability * 100, 2)
    return result_text, risk_level, confidence_percentage


def save_prediction_record(form_data, result_text, confidence_percentage, timestamp=None):
    """Store the prediction in SQLite and return a confirmation message for the UI."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    record_id = insert_prediction_record(form_data, result_text, confidence_percentage, timestamp)
    if record_id is None:
        raise RuntimeError("Could not save the prediction record to the database.")

    return timestamp, f"Prediction record saved successfully in the database (ID: {record_id})."

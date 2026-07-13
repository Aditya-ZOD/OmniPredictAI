import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request, send_from_directory
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from database import get_all_predictions, get_prediction_count, search_predictions_by_date
from predict import prepare_form_data, run_prediction, save_prediction_record


# Create the Flask application instance.
app = Flask(__name__)

# Set the base directory and model paths.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "heart_model.pkl"
SCALER_PATH = BASE_DIR / "model" / "scaler.pkl"
HISTORY_PATH = BASE_DIR / "prediction_history.csv"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


# Load the trained model and scaler from disk.
def load_model_artifacts():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


MODEL, SCALER = load_model_artifacts()


# Save each prediction into a CSV file for later review.
def save_prediction_history(form_data, result_text, confidence_percentage):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    history_row = {
        "timestamp": timestamp,
        "patient_name": form_data.get("patient_name", "").strip(),
        "age": form_data["age"],
        "sex": form_data["sex"],
        "cp": form_data["cp"],
        "trestbps": form_data["trestbps"],
        "chol": form_data["chol"],
        "fbs": form_data["fbs"],
        "restecg": form_data["restecg"],
        "thalach": form_data["thalach"],
        "exang": form_data["exang"],
        "oldpeak": form_data["oldpeak"],
        "slope": form_data["slope"],
        "ca": form_data["ca"],
        "thal": form_data["thal"],
        "prediction_result": result_text,
        "confidence_score": confidence_percentage,
    }

    if HISTORY_PATH.exists():
        history_df = pd.read_csv(HISTORY_PATH)
        history_df = pd.concat([history_df, pd.DataFrame([history_row])], ignore_index=True)
    else:
        history_df = pd.DataFrame([history_row])

    history_df.to_csv(HISTORY_PATH, index=False)


# Generate a PDF report for each prediction and save it in the reports folder.
def generate_pdf_report(form_data, result_text, confidence_percentage, risk_level, timestamp):
    report_path = REPORTS_DIR / f"prediction_report_{timestamp.replace(':', '-').replace(' ', '_')}.pdf"

    doc = SimpleDocTemplate(str(report_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Heart Disease Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Patient: {form_data.get('patient_name', '').strip()}", styles["Heading3"]))
    story.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
    story.append(Spacer(1, 12))

    patient_data = [
        ["Field", "Value"],
        ["Patient Name", str(form_data.get("patient_name", "").strip())],
        ["Age", str(form_data["age"])],
        ["Sex", str(form_data["sex"])],
        ["Chest Pain Type", str(form_data["cp"])],
        ["Resting Blood Pressure", str(form_data["trestbps"])],
        ["Cholesterol", str(form_data["chol"])],
        ["Fasting Blood Sugar", str(form_data["fbs"])],
        ["ECG Result", str(form_data["restecg"])],
        ["Maximum Heart Rate", str(form_data["thalach"])],
        ["Exercise Induced Angina", str(form_data["exang"])],
        ["ST Depression", str(form_data["oldpeak"])],
        ["Slope", str(form_data["slope"])],
        ["Number of Major Vessels", str(form_data["ca"])],
        ["Thalassemia", str(form_data["thal"])],
    ]

    table = Table(patient_data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), "#2F6F9F"),
                ("TEXTCOLOR", (0, 0), (-1, 0), "white"),
                ("GRID", (0, 0), (-1, -1), 0.5, "#BFBFBF"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
            ]
        )
    )

    story.append(Paragraph("Patient Data", styles["Heading2"]))
    story.append(table)
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Prediction Result: {result_text}", styles["Heading3"]))
    story.append(Paragraph(f"Confidence Score: {confidence_percentage}%", styles["Normal"]))
    story.append(Paragraph(f"Risk Level: {risk_level}", styles["Normal"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Health Suggestions", styles["Heading2"]))

    if risk_level == "High Risk":
        suggestions = [
            "Schedule a consultation with a healthcare professional as soon as possible.",
            "Monitor blood pressure, cholesterol, and heart rate regularly.",
            "Reduce sodium, avoid smoking, and maintain a heart-healthy diet.",
        ]
    elif risk_level == "Moderate Risk":
        suggestions = [
            "Increase daily physical activity and follow a balanced diet.",
            "Keep routine checkups and review your cardiovascular health with a doctor.",
            "Track symptoms such as chest pain, fatigue, or shortness of breath.",
        ]
    else:
        suggestions = [
            "Maintain your current healthy habits and continue regular exercise.",
            "Keep annual health screenings and monitor key vitals.",
            "Stay attentive to new symptoms and seek care if they appear.",
        ]

    for item in suggestions:
        story.append(Paragraph(f"• {item}", styles["Normal"]))

    doc.build(story)
    return report_path


# Define the expected feature names for the model input.
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


@app.route("/", methods=["GET"])
def home():
    """Render the home page with the prediction form."""
    return render_template("index.html")


@app.route("/download-report/<path:filename>", methods=["GET"])
def download_report(filename):
    """Serve a generated PDF report to the browser."""
    return send_from_directory(REPORTS_DIR, filename, as_attachment=True)


@app.route("/predict", methods=["POST"])
def predict():
    """Receive form data, preprocess it, and return the prediction result."""
    try:
        form_data = prepare_form_data(request.form)

        if not form_data.get("patient_name", "").strip():
            raise ValueError("Please enter the patient's name before submitting the form.")

        if None in form_data.values():
            raise ValueError("Please complete all fields before submitting the form.")

        print(f"Prediction requested for patient: {form_data['patient_name']}")
        result_text, risk_level, confidence_percentage = run_prediction(form_data, MODEL, SCALER)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save the prediction to the SQLite database and keep the CSV history file for compatibility.
        save_prediction_history(form_data, result_text, confidence_percentage)
        _, database_message = save_prediction_record(form_data, result_text, confidence_percentage, timestamp)
        print(f"Prediction saved for patient: {form_data['patient_name']} ({database_message})")
        report_path = generate_pdf_report(form_data, result_text, confidence_percentage, risk_level, timestamp)

        return render_template(
            "result.html",
            prediction=result_text,
            probability=confidence_percentage,
            risk_level=risk_level,
            report_path=report_path.name,
            database_message=database_message,
            patient_name=form_data["patient_name"],
        )

    except FileNotFoundError as exc:
        return render_template("result.html", error_message=str(exc)), 500
    except ValueError as exc:
        return render_template("result.html", error_message=str(exc)), 400
    except Exception as exc:
        return render_template("result.html", error_message="Prediction failed. Please try again."), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5005)

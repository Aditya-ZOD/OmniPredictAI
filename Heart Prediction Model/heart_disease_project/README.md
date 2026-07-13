# Heart Disease Prediction Project

## Project Overview
This project is a Flask-based web application that predicts the likelihood of heart disease using a trained machine learning model. It provides a modern, responsive interface for entering patient health details and receiving a prediction with confidence and risk information.

## Features
- Responsive medical dashboard UI
- Heart disease prediction form
- Model training and evaluation pipeline
- Prediction result with confidence score and risk level
- PDF report generation for each prediction
- Prediction history stored in CSV format
- Dark/light mode support

## Installation
1. Clone the project folder.
2. Navigate to the project directory.
3. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Train the model:
   ```bash
   python train_model.py
   ```
2. Start the Flask application:
   ```bash
   python app.py
   ```
3. Open the browser and go to:
   ```text
   http://127.0.0.1:5000/
   ```
4. Fill in the form and submit it to view the prediction result and download the PDF report.

## Screenshots
- Home page form
- Prediction result page
- PDF report example

## Project Structure
- app.py: Flask application entry point
- train_model.py: model training and evaluation
- templates/: HTML pages
- static/: CSS and JavaScript files
- model/: trained model and scaler artifacts
- reports/: generated charts and PDF reports

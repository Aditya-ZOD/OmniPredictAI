# OmniPredict AI

OmniPredict AI is a Flask-based predictive analytics dashboard for uploading CSV datasets, configuring target columns, preprocessing data, training models, making predictions, and exporting reports.

## Features
- Upload and preview CSV datasets
- Automatic problem detection and preprocessing
- Model training and best-model selection
- Prediction history and reporting
- Admin dashboard metrics
- Dark/light mode with persisted preference

## Installation
1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   python app.py
   ```
4. Open http://127.0.0.1:5000/

## Deployment Notes
- Set a strong SECRET_KEY environment variable.
- Use a production WSGI server such as Gunicorn.
- Example:
  ```bash
  gunicorn --bind 0.0.0.0:8000 app:app
  ```

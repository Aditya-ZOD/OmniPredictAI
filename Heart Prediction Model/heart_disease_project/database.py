import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Define the base directory for the project so the database file is created in the app folder.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "heart_disease.db"

# SQL statement used to create the prediction history table if it does not already exist.
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    patient_name TEXT NOT NULL,
    age REAL NOT NULL,
    sex REAL NOT NULL,
    cp REAL NOT NULL,
    trestbps REAL NOT NULL,
    chol REAL NOT NULL,
    fbs REAL NOT NULL,
    restecg REAL NOT NULL,
    thalach REAL NOT NULL,
    exang REAL NOT NULL,
    oldpeak REAL NOT NULL,
    slope REAL NOT NULL,
    ca REAL NOT NULL,
    thal REAL NOT NULL,
    prediction_result TEXT NOT NULL,
    confidence_score REAL NOT NULL
)
"""


def initialize_database() -> bool:
    """Create the database file and ensure the prediction history table includes the patient name column."""
    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.execute(CREATE_TABLE_SQL)
            columns = [row[1] for row in connection.execute("PRAGMA table_info(prediction_history)")]
            if "patient_name" not in columns:
                connection.execute("ALTER TABLE prediction_history ADD COLUMN patient_name TEXT NOT NULL DEFAULT ''")
            connection.commit()
        return True
    except sqlite3.Error as exc:
        print(f"Database initialization failed: {exc}")
        return False


def insert_prediction_record(
    form_data: Dict[str, Any],
    result_text: str,
    confidence_score: float,
    timestamp: Optional[str] = None,
) -> Optional[int]:
    """Insert a completed prediction record into the database and return the new row id."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        with sqlite3.connect(DB_PATH) as connection:
            cursor = connection.execute(
                """
                INSERT INTO prediction_history (
                    timestamp, patient_name, age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                    exang, oldpeak, slope, ca, thal, prediction_result, confidence_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    str(form_data.get("patient_name", "")).strip(),
                    form_data.get("age"),
                    form_data.get("sex"),
                    form_data.get("cp"),
                    form_data.get("trestbps"),
                    form_data.get("chol"),
                    form_data.get("fbs"),
                    form_data.get("restecg"),
                    form_data.get("thalach"),
                    form_data.get("exang"),
                    form_data.get("oldpeak"),
                    form_data.get("slope"),
                    form_data.get("ca"),
                    form_data.get("thal"),
                    result_text,
                    confidence_score,
                ),
            )
            connection.commit()
            return cursor.lastrowid
    except sqlite3.Error as exc:
        print(f"Failed to save prediction record: {exc}")
        return None


def get_all_predictions() -> List[Dict[str, Any]]:
    """Return every stored prediction record ordered from newest to oldest."""
    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM prediction_history ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        print(f"Failed to fetch predictions: {exc}")
        return []


def search_predictions_by_date(search_date: str) -> List[Dict[str, Any]]:
    """Return all predictions for a specific date using the YYYY-MM-DD format."""
    try:
        with sqlite3.connect(DB_PATH) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM prediction_history WHERE substr(timestamp, 1, 10) = ? ORDER BY id DESC",
                (search_date,),
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        print(f"Failed to search predictions by date: {exc}")
        return []


def get_prediction_count() -> int:
    """Return the total number of stored prediction records."""
    try:
        with sqlite3.connect(DB_PATH) as connection:
            row = connection.execute("SELECT COUNT(*) AS total FROM prediction_history").fetchone()
            return int(row[0]) if row else 0
    except sqlite3.Error as exc:
        print(f"Failed to count predictions: {exc}")
        return 0


# Ensure the database and table are created when this module is imported.
initialize_database()

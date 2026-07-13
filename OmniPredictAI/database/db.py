import os
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), 'omnipredict.db')
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.sql')

def get_db_connection():
    """Establishes a connection to the SQLite database with Row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database using the schema.sql file."""
    if not os.path.exists(SCHEMA_PATH):
        raise FileNotFoundError(f"Schema file not found at {SCHEMA_PATH}")

    conn = get_db_connection()
    try:
        with open(SCHEMA_PATH, 'r') as f:
            conn.executescript(f.read())
        conn.commit()
    finally:
        conn.close()

# ─── User Queries ─────────────────────────────────────────────────────────────

def create_user(email, password_hash, full_name):
    """Inserts a new user record into the database.

    Returns:
        int: The ID of the newly created user, or None if the email already exists.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (email, password_hash, full_name) VALUES (?, ?, ?)",
            (email.lower().strip(), password_hash, full_name.strip())
        )
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()

def get_user_by_email(email):
    """Retrieves a user by their email address."""
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email.lower().strip(),)
        ).fetchone()
    finally:
        conn.close()

def get_user_by_id(user_id):
    """Retrieves a user by their unique database ID."""
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
    finally:
        conn.close()

# ─── Dataset Queries ──────────────────────────────────────────────────────────

def save_dataset(user_id, filename, original_name, file_path, num_rows, num_cols, file_size):
    """Inserts a dataset metadata record.

    Returns:
        int: The new dataset ID.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """INSERT INTO datasets
               (user_id, filename, original_name, file_path, num_rows, num_cols, file_size)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, filename, original_name, file_path, num_rows, num_cols, file_size)
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()

def get_datasets_by_user(user_id):
    """Returns all datasets uploaded by a specific user, newest first."""
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,)
        ).fetchall()
    finally:
        conn.close()

def get_dataset_by_id(dataset_id, user_id):
    """Returns a single dataset record owned by the given user."""
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM datasets WHERE id = ? AND user_id = ?",
            (dataset_id, user_id)
        ).fetchone()
    finally:
        conn.close()

def delete_dataset(dataset_id, user_id):
    """Deletes a dataset record from the database."""
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM datasets WHERE id = ? AND user_id = ?",
            (dataset_id, user_id)
        )
        conn.commit()
    finally:
        conn.close()

# ─── Model Config Queries ─────────────────────────────────────────────────────

def save_model_config(dataset_id, user_id, target_column, problem_type, excluded_columns_json):
    """Upserts a model configuration for a given dataset."""
    conn = get_db_connection()
    try:
        conn.execute(
            """INSERT INTO model_configs
               (dataset_id, user_id, target_column, problem_type, excluded_columns)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(dataset_id) DO UPDATE SET
                   target_column     = excluded.target_column,
                   problem_type      = excluded.problem_type,
                   excluded_columns  = excluded.excluded_columns,
                   configured_at     = CURRENT_TIMESTAMP""",
            (dataset_id, user_id, target_column, problem_type, excluded_columns_json)
        )
        conn.commit()
    finally:
        conn.close()

def get_model_config(dataset_id):
    """Returns the model config for a dataset, or None."""
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM model_configs WHERE dataset_id = ?",
            (dataset_id,)
        ).fetchone()
    finally:
        conn.close()

def delete_model_config(dataset_id, user_id):
    """Removes a model config record."""
    conn = get_db_connection()
    try:
        conn.execute(
            "DELETE FROM model_configs WHERE dataset_id = ? AND user_id = ?",
            (dataset_id, user_id)
        )
        conn.commit()
    finally:
        conn.close()

# ─── Preprocessing Result Queries ─────────────────────────────────────────────

def save_preprocessing_result(dataset_id, user_id, summary_json,
                               x_train_path, x_test_path,
                               y_train_path, y_test_path,
                               scaler_path, encoder_path, training_result_path):
    """Upserts preprocessing results for a dataset."""
    conn = get_db_connection()
    try:
        conn.execute(
            """INSERT INTO preprocessing_results
               (dataset_id, user_id, summary_json,
                x_train_path, x_test_path, y_train_path, y_test_path,
                scaler_path, encoder_path, training_result_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(dataset_id) DO UPDATE SET
                   summary_json       = excluded.summary_json,
                   x_train_path       = excluded.x_train_path,
                   x_test_path        = excluded.x_test_path,
                   y_train_path       = excluded.y_train_path,
                   y_test_path        = excluded.y_test_path,
                   scaler_path        = excluded.scaler_path,
                   encoder_path       = excluded.encoder_path,
                   training_result_path = excluded.training_result_path,
                   preprocessed_at    = CURRENT_TIMESTAMP""",
            (dataset_id, user_id, summary_json,
             x_train_path, x_test_path, y_train_path, y_test_path,
             scaler_path, encoder_path, training_result_path)
        )
        conn.commit()
    finally:
        conn.close()

def get_preprocessing_result(dataset_id):
    """Returns the preprocessing result record for a dataset, or None."""
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM preprocessing_results WHERE dataset_id = ?",
            (dataset_id,)
        ).fetchone()
    finally:
        conn.close()


def save_prediction_history(user_id, dataset_id, dataset_name, user_name, selected_model, prediction_result, confidence_score):
    """Stores a prediction event for the current user."""
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO prediction_history (user_id, dataset_id, dataset_name, user_name, selected_model, prediction_result, confidence_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, dataset_id, dataset_name, user_name, selected_model, prediction_result, confidence_score)
        )
        conn.commit()
    finally:
        conn.close()


def get_prediction_history(user_id):
    """Returns all prediction history rows for a user, newest first."""
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT * FROM prediction_history WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
    finally:
        conn.close()


def get_admin_dashboard_stats():
    """Returns aggregate stats for the admin dashboard."""
    conn = get_db_connection()
    try:
        total_users = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()['count']
        total_datasets = conn.execute("SELECT COUNT(*) AS count FROM datasets").fetchone()['count']
        total_predictions = conn.execute("SELECT COUNT(*) AS count FROM prediction_history").fetchone()['count']

        algorithm_rows = conn.execute(
            "SELECT selected_model AS name, COUNT(*) AS count FROM prediction_history GROUP BY selected_model ORDER BY count DESC"
        ).fetchall()
        most_used_algorithm = algorithm_rows[0]['name'] if algorithm_rows else 'N/A'

        accuracy_rows = conn.execute(
            "SELECT selected_model AS name, AVG(confidence_score) AS avg_score FROM prediction_history WHERE confidence_score IS NOT NULL GROUP BY selected_model"
        ).fetchall()
        best_average_accuracy = None
        if accuracy_rows:
            best_average_accuracy = max(row['avg_score'] for row in accuracy_rows) if accuracy_rows else None

        return {
            'total_users': total_users,
            'total_datasets': total_datasets,
            'total_predictions': total_predictions,
            'most_used_algorithm': most_used_algorithm,
            'best_average_accuracy': round(best_average_accuracy, 4) if best_average_accuracy is not None else None,
        }
    finally:
        conn.close()

# Auto-initialize database on module load
init_db()



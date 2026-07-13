import os
import uuid
import json
import pickle
from functools import wraps
from datetime import timedelta
from io import BytesIO

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import pandas as pd
from flask import (Flask, render_template, redirect, url_for,
                   request, session, g, flash, jsonify, send_file, send_from_directory)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from database import db
import preprocessing as pp_engine

# ─── App Configuration ────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'omnipredict_ai_secure_secret_key_12345')
app.config['DEBUG'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# ─── Custom Jinja2 Filters ────────────────────────────────────────────────────
@app.template_filter('enumerate')
def jinja_enumerate(iterable):
    return enumerate(iterable)

@app.template_filter('unique')
def jinja_unique(iterable):
    seen = set()
    result = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_MB = 50  # MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 ** 2):.1f} MB"


def load_pickle(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def validate_dataset_file(file_path):
    """Validate uploaded CSV data and raise a clear error for unusable datasets."""
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f'Unable to read the CSV file: {exc}') from exc

    if df.empty:
        raise ValueError('The dataset is empty.')

    if df.shape[0] < 3:
        raise ValueError('The dataset must contain at least 3 rows.')

    if df.shape[1] < 2:
        raise ValueError('The dataset must contain at least 2 columns.')

    if df.isnull().all().any():
        raise ValueError('At least one column contains only missing values.')

    return df


def fix_visualization_paths(summary):
    """Normalize visualization chart file paths for templates and local routing."""
    if summary and 'visualizations' in summary:
        fixed_visuals = {}
        for key, filepath in summary['visualizations'].items():
            if filepath:
                filename = os.path.basename(filepath)
                fixed_visuals[key] = f"reports/{filename}"
        summary['visualizations'] = fixed_visuals
    return summary


def build_report_pdf(report_data):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setTitle('OmniPredict AI Report')
    pdf.setFont('Helvetica-Bold', 16)
    pdf.drawString(40, height - 40, 'OmniPredict AI Report')
    pdf.setFont('Helvetica', 11)
    pdf.drawString(40, height - 70, f"Dataset: {report_data['dataset_name']}")
    pdf.drawString(40, height - 90, f"Best Model: {report_data['best_model_name']}")

    y = height - 130
    pdf.setFont('Helvetica-Bold', 12)
    pdf.drawString(40, y, 'Dataset Summary')
    pdf.setFont('Helvetica', 10)
    y -= 15
    pdf.drawString(40, y, f"Rows: {report_data['num_rows']}")
    y -= 12
    pdf.drawString(40, y, f"Columns: {report_data['num_cols']}")
    y -= 12
    pdf.drawString(40, y, f"Target: {report_data['target_column']}")

    y -= 25
    pdf.setFont('Helvetica-Bold', 12)
    pdf.drawString(40, y, 'Model Metrics')
    pdf.setFont('Helvetica', 10)
    y -= 15
    for key, value in report_data['metrics'].items():
        pdf.drawString(40, y, f"- {key}: {value}")
        y -= 12

    y -= 10
    pdf.setFont('Helvetica-Bold', 12)
    pdf.drawString(40, y, 'Recent Predictions')
    pdf.setFont('Helvetica', 10)
    y -= 15
    for item in report_data['history'][:5]:
        pdf.drawString(40, y, f"- {item['prediction_result']} ({item['selected_model']})")
        y -= 12

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer

# ─── Session / Auth ───────────────────────────────────────────────────────────

@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')
    g.user = db.get_user_by_id(user_id) if user_id else None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if g.user is None:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ─── Public Routes ────────────────────────────────────────────────────────────

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html', page_name='home')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email    = request.form.get('email', '')
        password = request.form.get('password', '')
        remember = request.form.get('remember') == 'on'

        if not email or not password:
            flash('Please provide both email and password.', 'danger')
            return render_template('login.html', page_name='login')

        user = db.get_user_by_email(email)
        if user and check_password_hash(user['password_hash'], password):
            session.clear()
            session['user_id']   = user['id']
            session['user_name'] = user['full_name']
            session.permanent    = remember
            flash(f'Welcome back, {user["full_name"]}!', 'success')
            return redirect(url_for('dashboard'))

        flash('Invalid email or password. Please try again.', 'danger')

    return render_template('login.html', page_name='login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if g.user:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        full_name = request.form.get('fullName', '').strip()
        email     = request.form.get('email', '').strip()
        password  = request.form.get('password', '')

        if not full_name or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html', page_name='register')

        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('register.html', page_name='register')

        if db.get_user_by_email(email):
            flash('Email address already registered.', 'danger')
            return render_template('register.html', page_name='register')

        new_id = db.create_user(email, generate_password_hash(password), full_name)
        if new_id:
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

        flash('An error occurred during registration. Please try again.', 'danger')

    return render_template('register.html', page_name='register')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))

# ─── Dashboard ────────────────────────────────────────────────────────────────

@app.route('/dashboard')
@login_required
def dashboard():
    raw = db.get_datasets_by_user(g.user['id'])
    datasets = []
    for d in raw:
        row = dict(d)
        row['size_display'] = format_size(d['file_size'] or 0)
        cfg = db.get_model_config(d['id'])
        row['is_configured'] = cfg is not None
        row['target_column'] = cfg['target_column'] if cfg else None
        pre = db.get_preprocessing_result(d['id'])
        row['is_preprocessed'] = pre is not None
        datasets.append(row)
    latest_result = None
    latest_summary = None
    for dataset in datasets:
        pre = db.get_preprocessing_result(dataset['id'])
        if pre:
            latest_result = pre
            latest_summary = json.loads(pre['summary_json']) if pre['summary_json'] else None
            if latest_summary:
                latest_summary = fix_visualization_paths(latest_summary)
            break

    return render_template(
        'dashboard.html',
        page_name='dashboard',
        datasets=datasets,
        latest_summary=latest_summary,
    )


@app.route('/admin')
@login_required
def admin_dashboard():
    stats = db.get_admin_dashboard_stats()
    return render_template('admin_dashboard.html', page_name='admin', stats=stats)

# ─── Upload ───────────────────────────────────────────────────────────────────

@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'file' not in request.files:
        flash('No file selected.', 'danger')
        return redirect(url_for('dashboard'))

    f = request.files['file']
    if f.filename == '':
        flash('No file selected.', 'danger')
        return redirect(url_for('dashboard'))

    if not allowed_file(f.filename):
        flash('Only CSV files are supported.', 'danger')
        return redirect(url_for('dashboard'))

    # Save with a unique name to avoid collisions
    original_name = secure_filename(f.filename)
    unique_name   = f"{uuid.uuid4().hex}_{original_name}"
    file_path     = os.path.join(UPLOAD_FOLDER, unique_name)
    f.save(file_path)

    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_MB * 1024 * 1024:
        os.remove(file_path)
        flash(f'File exceeds the {MAX_FILE_MB} MB limit.', 'danger')
        return redirect(url_for('dashboard'))

    try:
        df_full = validate_dataset_file(file_path)
        num_rows, num_cols = df_full.shape
    except ValueError as e:
        os.remove(file_path)
        flash(str(e), 'danger')
        return redirect(url_for('dashboard'))
    except Exception as e:
        os.remove(file_path)
        flash(f'Unexpected upload error: {e}', 'danger')
        return redirect(url_for('dashboard'))

    db.save_dataset(
        user_id       = g.user['id'],
        filename      = unique_name,
        original_name = original_name,
        file_path     = file_path,
        num_rows      = num_rows,
        num_cols      = num_cols,
        file_size     = file_size,
    )

    flash(f'"{original_name}" uploaded successfully — {num_rows:,} rows × {num_cols} columns.', 'success')
    return redirect(url_for('dashboard'))

# ─── Dataset Preview ──────────────────────────────────────────────────────────

@app.route('/dataset/<int:dataset_id>')
@login_required
def dataset_preview(dataset_id):
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dashboard'))

    try:
        df = validate_dataset_file(record['file_path'])
    except ValueError as e:
        flash(str(e), 'danger')
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f'Could not read dataset: {e}', 'danger')
        return redirect(url_for('dashboard'))

    preview_rows   = df.head(10).to_dict(orient='records')
    columns        = list(df.columns)
    dtypes         = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing        = df.isnull().sum().to_dict()
    missing_pct    = {col: round(cnt / len(df) * 100, 1) if len(df) else 0
                      for col, cnt in missing.items()}
    num_rows, num_cols = df.shape

    return render_template(
        'dataset_preview.html',
        page_name     = 'dashboard',
        record        = dict(record),
        preview_rows  = preview_rows,
        columns       = columns,
        dtypes        = dtypes,
        missing       = missing,
        missing_pct   = missing_pct,
        num_rows      = num_rows,
        num_cols      = num_cols,
        size_display  = format_size(record['file_size'] or 0),
    )

# ─── Delete Dataset ───────────────────────────────────────────────────────────

@app.route('/dataset/<int:dataset_id>/delete', methods=['POST'])
@login_required
def delete_dataset(dataset_id):
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if record:
        try:
            os.remove(record['file_path'])
        except FileNotFoundError:
            pass
        db.delete_dataset(dataset_id, g.user['id'])
        flash('Dataset deleted.', 'info')
    else:
        flash('Dataset not found.', 'danger')
    return redirect(url_for('dashboard'))

# ─── Configure Target Column ─────────────────────────────────────────────────

@app.route('/dataset/<int:dataset_id>/configure', methods=['GET', 'POST'])
@login_required
def configure_target(dataset_id):
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dashboard'))

    try:
        df = validate_dataset_file(record['file_path'])
    except ValueError as e:
        flash(str(e), 'danger')
        return redirect(url_for('dashboard'))
    except Exception as e:
        flash(f'Could not read dataset: {e}', 'danger')
        return redirect(url_for('dashboard'))

    columns  = list(df.columns)
    dtypes   = {col: str(dtype) for col, dtype in df.dtypes.items()}
    existing = db.get_model_config(dataset_id)

    # ── POST: Save configuration ─────────────────────────────────────────────
    if request.method == 'POST':
        target   = request.form.get('target_column', '').strip()
        ptype    = request.form.get('problem_type', 'auto')
        excluded = request.form.getlist('excluded_columns')

        # Validation
        if not target:
            flash('Please select a target column.', 'danger')
        elif target not in columns:
            flash(f'Column "{target}" does not exist in the dataset.', 'danger')
        elif target in excluded:
            flash('Target column cannot also be excluded.', 'danger')
        else:
            detected_problem_type = pp_engine.detect_problem_type(df[target])
            effective_problem_type = detected_problem_type if ptype == 'auto' else ptype
            db.save_model_config(
                dataset_id        = dataset_id,
                user_id           = g.user['id'],
                target_column     = target,
                problem_type      = effective_problem_type,
                excluded_columns_json = json.dumps(excluded),
            )
            flash(f'Target column set to "{target}". Detected as {effective_problem_type.title()}.', 'success')
            return redirect(url_for('configure_target', dataset_id=dataset_id))

    # ── GET: Build page data ─────────────────────────────────────────────────
    # Column statistics for the summary panel
    col_stats = {}
    for col in columns:
        s = df[col]
        stat = {
            'dtype':   dtypes[col],
            'missing': int(s.isnull().sum()),
            'missing_pct': round(s.isnull().sum() / len(df) * 100, 1) if len(df) else 0,
            'unique':  int(s.nunique()),
        }
        if pd.api.types.is_numeric_dtype(s):
            stat.update({
                'is_numeric': True,
                'mean':  round(float(s.mean()), 4) if not s.isnull().all() else None,
                'std':   round(float(s.std()),  4) if not s.isnull().all() else None,
                'min':   round(float(s.min()),  4) if not s.isnull().all() else None,
                'max':   round(float(s.max()),  4) if not s.isnull().all() else None,
            })
        else:
            stat['is_numeric'] = False
            top = s.value_counts().head(3).to_dict()
            stat['top_values'] = top
        col_stats[col] = stat

    saved_excluded = json.loads(existing['excluded_columns']) if existing else []

    return render_template(
        'configure_target.html',
        page_name      = 'dashboard',
        record         = dict(record),
        columns        = columns,
        dtypes         = dtypes,
        col_stats      = col_stats,
        existing       = dict(existing) if existing else None,
        saved_excluded = saved_excluded,
        num_rows       = len(df),
        num_cols       = len(columns),
        size_display   = format_size(record['file_size'] or 0),
    )

# ─── Preprocess Dataset ─────────────────────────────────────────────────────

@app.route('/dataset/<int:dataset_id>/preprocess', methods=['GET', 'POST'])
@login_required
def preprocess(dataset_id):
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dashboard'))

    config = db.get_model_config(dataset_id)
    if not config:
        flash('Please configure a target column before preprocessing.', 'warning')
        return redirect(url_for('configure_target', dataset_id=dataset_id))

    existing_result = db.get_preprocessing_result(dataset_id)
    excluded = json.loads(config['excluded_columns'])

    if request.method == 'POST':
        test_size         = float(request.form.get('test_size', 0.2))
        scale_method      = request.form.get('scale_method', 'standard')
        missing_strategy  = request.form.get('missing_strategy', 'auto')

        # Save artifacts per-dataset in uploads/<dataset_id>/
        save_dir = os.path.join(UPLOAD_FOLDER, f'dataset_{dataset_id}')

        try:
            result = pp_engine.run_preprocessing(
                file_path        = record['file_path'],
                target_column    = config['target_column'],
                excluded_columns = excluded,
                test_size        = test_size,
                scale_method     = scale_method,
                missing_strategy = missing_strategy,
                save_dir         = save_dir,
            )
        except Exception as e:
            flash(f'Preprocessing failed: {e}', 'danger')
            return redirect(url_for('preprocess', dataset_id=dataset_id))

        summary   = result['summary']
        artifacts = result['artifacts']

        if config['problem_type'] == 'auto':
            db.save_model_config(
                dataset_id        = dataset_id,
                user_id           = g.user['id'],
                target_column     = config['target_column'],
                problem_type      = summary.get('problem_type', 'classification'),
                excluded_columns_json = json.dumps(excluded),
            )

        db.save_preprocessing_result(
            dataset_id   = dataset_id,
            user_id      = g.user['id'],
            summary_json = json.dumps(summary),
            x_train_path = artifacts.get('x_train_path'),
            x_test_path  = artifacts.get('x_test_path'),
            y_train_path = artifacts.get('y_train_path'),
            y_test_path  = artifacts.get('y_test_path'),
            scaler_path  = artifacts.get('scaler_path'),
            encoder_path = artifacts.get('encoder_path'),
            training_result_path = artifacts.get('training_result_path'),
        )

        flash('Preprocessing completed successfully!', 'success')
        return redirect(url_for('preprocess', dataset_id=dataset_id))

    # GET – show options form + results if they exist
    preprocess_summary = None
    if existing_result:
        preprocess_summary = json.loads(existing_result['summary_json'])

    return render_template(
        'preprocess.html',
        page_name         = 'dashboard',
        record            = dict(record),
        config            = dict(config),
        existing_result   = dict(existing_result) if existing_result else None,
        preprocess_summary= preprocess_summary,
        size_display      = format_size(record['file_size'] or 0),
    )

# ─── Prediction ─────────────────────────────────────────────────────────────

@app.route('/dataset/<int:dataset_id>/predict', methods=['GET', 'POST'])
@login_required
def predict_dataset(dataset_id):
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dashboard'))

    preprocessing_result = db.get_preprocessing_result(dataset_id)
    if not preprocessing_result:
        flash('Please run preprocessing before making predictions.', 'warning')
        return redirect(url_for('preprocess', dataset_id=dataset_id))

    summary = json.loads(preprocessing_result['summary_json']) if preprocessing_result['summary_json'] else {}
    feature_columns = summary.get('feature_columns', [])
    problem_type = summary.get('problem_type', 'classification')
    target_column = summary.get('target_column')

    model_path = summary.get('best_model', {}).get('path')
    model = load_pickle(model_path) if model_path else None

    scaler = None
    if preprocessing_result['scaler_path']:
        scaler = load_pickle(preprocessing_result['scaler_path'])

    encoder_bundle = None
    if preprocessing_result['encoder_path']:
        encoder_bundle = load_pickle(preprocessing_result['encoder_path'])

    prediction = None
    confidence = None
    feature_values = {}

    if request.method == 'POST':
        if not feature_columns:
            flash('No feature columns were found for this dataset.', 'warning')
            return redirect(url_for('predict_dataset', dataset_id=dataset_id))

        for feature in feature_columns:
            raw_value = request.form.get(feature, '').strip()
            if raw_value == '':
                flash(f'Please provide a value for {feature}.', 'warning')
                return redirect(url_for('predict_dataset', dataset_id=dataset_id))
            try:
                feature_values[feature] = float(raw_value)
            except ValueError:
                flash(f'Value for {feature} must be numeric.', 'danger')
                return redirect(url_for('predict_dataset', dataset_id=dataset_id))

        if model is None:
            flash('The trained model could not be loaded.', 'danger')
            return redirect(url_for('predict_dataset', dataset_id=dataset_id))

        # ── Compute prediction first, then persist to history ───────────────
        input_df = pd.DataFrame([feature_values])
        if scaler is not None and hasattr(scaler, 'transform'):
            input_df = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

        prediction_raw = model.predict(input_df)[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)[0]
            best_idx = max(range(len(probabilities)), key=lambda i: probabilities[i])
            confidence = float(probabilities[best_idx])
            if encoder_bundle and encoder_bundle.get('target_encoder') is not None:
                classes = list(encoder_bundle['target_encoder'].classes_)
                prediction = classes[best_idx] if best_idx < len(classes) else prediction_raw
            else:
                prediction = prediction_raw
        else:
            prediction = prediction_raw

        # Save history only after prediction is known
        selected_model = summary.get('best_model', {}).get('name', 'unknown_model')
        db.save_prediction_history(
            user_id=g.user['id'],
            dataset_id=dataset_id,
            dataset_name=record['original_name'],
            user_name=g.user['full_name'],
            selected_model=selected_model,
            prediction_result=str(prediction),
            confidence_score=confidence,
        )

    # Extra context for richer UI
    best_model_info = summary.get('best_model', {})
    model_scores    = summary.get('model_scores', {})
    all_probabilities = {}
    if request.method == 'POST' and model is not None and feature_values:
        try:
            _input_df = pd.DataFrame([feature_values])
            if scaler is not None and hasattr(scaler, 'transform'):
                _input_df = pd.DataFrame(scaler.transform(_input_df), columns=feature_columns)
            if hasattr(model, 'predict_proba') and encoder_bundle and encoder_bundle.get('target_encoder'):
                _proba = model.predict_proba(_input_df)[0]
                _classes = list(encoder_bundle['target_encoder'].classes_)
                all_probabilities = dict(zip(_classes, [round(float(p)*100, 1) for p in _proba]))
        except Exception:
            pass

    return render_template(
        'predict.html',
        page_name='dashboard',
        record=dict(record),
        feature_columns=feature_columns,
        problem_type=problem_type,
        target_column=target_column,
        feature_values=feature_values,
        prediction=prediction,
        confidence=confidence,
        size_display=format_size(record['file_size'] or 0),
        model_scores=model_scores,
        best_model_name=best_model_info.get('name', ''),
        best_model_accuracy=best_model_info.get('metrics', {}).get('accuracy') or best_model_info.get('metrics', {}).get('r2'),
        all_probabilities=all_probabilities,
    )

# ─── Reports ──────────────────────────────────────────────────────────────────

@app.route('/reports/<path:filename>')
@login_required
def serve_report_image(filename):
    reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    return send_from_directory(reports_dir, filename)

@app.route('/reports')
@login_required
def reports():
    datasets = db.get_datasets_by_user(g.user['id'])
    latest_dataset = None
    report_data = None
    for dataset in datasets:
        pre = db.get_preprocessing_result(dataset['id'])
        if pre:
            latest_dataset = dataset
            summary = json.loads(pre['summary_json']) if pre['summary_json'] else {}
            summary = fix_visualization_paths(summary)
            report_data = {
                'dataset_name': dataset['original_name'],
                'num_rows': dataset['num_rows'],
                'num_cols': dataset['num_cols'],
                'target_column': summary.get('target_column'),
                'best_model_name': summary.get('best_model', {}).get('name', 'N/A'),
                'metrics': summary.get('model_scores', {}),
                'visualizations': summary.get('visualizations', {}),
                'history': [
                    {
                        'prediction_result': row['prediction_result'],
                        'selected_model': row['selected_model'],
                        'created_at': row['created_at'],
                    }
                    for row in db.get_prediction_history(g.user['id'])[:5]
                ],
            }
            break

    return render_template('reports.html', page_name='reports', report_data=report_data, latest_dataset=latest_dataset)

@app.route('/reports/download')
@login_required
def download_report():
    datasets = db.get_datasets_by_user(g.user['id'])
    for dataset in datasets:
        pre = db.get_preprocessing_result(dataset['id'])
        if pre:
            summary = json.loads(pre['summary_json']) if pre['summary_json'] else {}
            report_data = {
                'dataset_name': dataset['original_name'],
                'num_rows': dataset['num_rows'],
                'num_cols': dataset['num_cols'],
                'target_column': summary.get('target_column'),
                'best_model_name': summary.get('best_model', {}).get('name', 'N/A'),
                'metrics': summary.get('model_scores', {}),
                'history': [
                    {
                        'prediction_result': row['prediction_result'],
                        'selected_model': row['selected_model'],
                        'created_at': row['created_at'],
                    }
                    for row in db.get_prediction_history(g.user['id'])[:5]
                ],
            }
            pdf_buffer = build_report_pdf(report_data)
            return send_file(pdf_buffer, download_name=f"{dataset['original_name']}_report.pdf", as_attachment=True, mimetype='application/pdf')

    flash('No report data available yet.', 'warning')
    return redirect(url_for('reports'))

@app.route('/history')
@login_required
def prediction_history():
    history = db.get_prediction_history(g.user['id'])
    return render_template('prediction_history.html', page_name='dashboard', history=history)

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Development entry point. For deployment, use a production WSGI server such as Gunicorn.
    app.run(host='127.0.0.1', port=5000)

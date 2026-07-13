"""
OmniPredict AI – Automatic Data Preprocessing Engine
Performs the full preprocessing pipeline and returns a detailed summary.
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')


def detect_problem_type(target_series: pd.Series) -> str:
    """Infer whether the target is a classification or regression problem.

    Integers with 10 or fewer unique values (e.g. binary labels 0/1 or
    multi-class codes) are treated as classification.  Continuous floats
    are treated as regression.
    """
    if target_series is None:
        return 'classification'

    s = target_series.dropna()
    if s.empty:
        return 'classification'

    if pd.api.types.is_numeric_dtype(s):
        # Low-cardinality integers → classification (e.g. 0/1 labels)
        if pd.api.types.is_integer_dtype(s) and s.nunique() <= 10:
            return 'classification'
        return 'regression'

    return 'classification'


def train_models(X: pd.DataFrame, y: pd.Series, problem_type: str,
                 X_test: pd.DataFrame = None, y_test: pd.Series = None) -> dict:
    """Train a suite of models for classification or regression and return scores.

    If X_test / y_test are provided the models are *evaluated* on the held-out
    test set, giving honest generalisation metrics.  When they are absent
    (e.g. in unit tests) the training data is used as a fallback.
    """
    n_est = 50 if len(X) > 20000 else 100
    if problem_type == 'regression':
        models = {
            'linear_regression': LinearRegression(),
            'decision_tree_regressor': DecisionTreeRegressor(random_state=42),
            'random_forest_regressor': RandomForestRegressor(n_estimators=n_est, random_state=42),
            'gradient_boosting_regressor': GradientBoostingRegressor(n_estimators=n_est, random_state=42),
        }
    else:
        models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=n_est, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=3),
        }
        # Skip SVC for large datasets because of O(N^3) time complexity
        if len(X) <= 15000:
            models['svm'] = SVC(random_state=42)

    # Use test split for evaluation when available; fall back to train data
    X_eval = X_test if X_test is not None else X
    y_eval = y_test if y_test is not None else y

    trained_models = {}
    scores = {}
    for name, model in models.items():
        model.fit(X, y)
        preds = model.predict(X_eval)
        if problem_type == 'regression':
            scores[name] = float(r2_score(y_eval, preds))
        else:
            scores[name] = float(accuracy_score(y_eval, preds))
        trained_models[name] = {
            'model': model,
            'predictions': preds,
            'y_eval': y_eval,
        }

    return {
        'problem_type': problem_type,
        'trained_models': trained_models,
        'models': list(models.keys()),
        'scores': scores,
    }


def _save_chart(fig, save_path: str) -> str:
    """Save a matplotlib figure to disk and return the path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def generate_visualizations(df: pd.DataFrame, y: pd.Series, summary: dict, save_dir: str) -> dict:
    """Generate summary charts for the dataset and trained models."""
    charts = {}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        corr = numeric_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap')
        charts['correlation_heatmap'] = _save_chart(fig, os.path.join(save_dir, 'correlation_heatmap.png'))

    # Missing values chart
    missing = df.isnull().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        missing.plot(kind='bar', ax=ax, color='tomato')
        ax.set_title('Missing Values by Column')
        ax.set_ylabel('Missing Count')
        charts['missing_values'] = _save_chart(fig, os.path.join(save_dir, 'missing_values.png'))

    # Target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    if pd.api.types.is_numeric_dtype(y):
        sns.histplot(y.dropna(), bins=20, ax=ax)
        ax.set_title('Target Distribution')
    else:
        y.value_counts().plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Target Distribution')
    charts['target_distribution'] = _save_chart(fig, os.path.join(save_dir, 'target_distribution.png'))

    # Feature importance graph
    if not numeric_df.empty:
        feature_names = [c for c in numeric_df.columns if c != y.name]
        if feature_names:
            fig, ax = plt.subplots(figsize=(6, 4))
            importance = pd.Series(np.random.rand(len(feature_names)), index=feature_names)
            importance.sort_values(ascending=False).plot(kind='bar', ax=ax, color='mediumseagreen')
            ax.set_title('Feature Importance (placeholder)')
            charts['feature_importance'] = _save_chart(fig, os.path.join(save_dir, 'feature_importance.png'))

    # Model accuracy comparison graph
    scores = summary.get('model_scores', {})
    if scores:
        fig, ax = plt.subplots(figsize=(6, 4))
        pd.Series(scores).sort_values(ascending=False).plot(kind='bar', ax=ax, color='royalblue')
        ax.set_title('Model Score Comparison')
        ax.set_ylabel('Score')
        charts['model_accuracy'] = _save_chart(fig, os.path.join(save_dir, 'model_accuracy.png'))

    return charts


def select_best_model(X: pd.DataFrame, y: pd.Series, problem_type: str,
                      save_dir: str = None,
                      X_test: pd.DataFrame = None,
                      y_test: pd.Series = None) -> dict:
    """Train models, compare metrics on the test split, select the best, and optionally save it."""
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'models')

    training_result = train_models(X, y, problem_type, X_test=X_test, y_test=y_test)
    metrics_by_model = {}

    if problem_type == 'regression':
        for name, info in training_result['trained_models'].items():
            preds  = info['predictions']
            y_eval = info['y_eval']
            metrics_by_model[name] = {
                'mae':  float(mean_absolute_error(y_eval, preds)),
                'mse':  float(mean_squared_error(y_eval, preds)),
                'rmse': float(np.sqrt(mean_squared_error(y_eval, preds))),
                'r2':   float(r2_score(y_eval, preds)),
            }
        best_name = min(metrics_by_model, key=lambda n: (metrics_by_model[n]['rmse'], -metrics_by_model[n]['r2']))
        best_metric = 'rmse'
        sort_ascending = True
    else:
        for name, info in training_result['trained_models'].items():
            preds  = info['predictions']
            y_eval = info['y_eval']
            # Use 'weighted' averaging so multi-class targets don't raise ValueError
            metrics_by_model[name] = {
                'accuracy':  float(accuracy_score(y_eval, preds)),
                'precision': float(precision_score(y_eval, preds, average='weighted', zero_division=0)),
                'recall':    float(recall_score(y_eval, preds, average='weighted', zero_division=0)),
                'f1':        float(f1_score(y_eval, preds, average='weighted', zero_division=0)),
            }
        best_name = max(metrics_by_model, key=lambda n: (metrics_by_model[n]['f1'], metrics_by_model[n]['accuracy']))
        best_metric = 'f1'
        sort_ascending = False

    best_model = training_result['trained_models'][best_name]['model']
    best_model_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, f'{best_name}.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)
    else:
        best_model_path = None

    return {
        'problem_type': problem_type,
        'best_model_name': best_name,
        'best_model': best_model,
        'best_model_path': best_model_path,
        'metrics': metrics_by_model,
        'best_metric': best_metric,
        'sort_ascending': sort_ascending,
    }


# ─── Main Pipeline ─────────────────────────────────────────────────────────────

def run_preprocessing(
    file_path: str,
    target_column: str,
    excluded_columns: list,
    test_size: float = 0.2,
    scale_method: str = 'standard',
    missing_strategy: str = 'auto',
    save_dir: str = None,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns a dict containing:
      - summary:  human-readable step-by-step log
      - artifacts: paths to saved split arrays and transformers
    """
    summary = {
        'steps': [],
        'original_shape': None,
        'final_shape': None,
        'duplicates_removed': 0,
        'missing_handled': {},
        'encoded_columns': {},
        'scaled_columns': [],
        'scale_method': scale_method,
        'test_size': test_size,
        'train_samples': 0,
        'test_samples': 0,
        'target_column': target_column,
        'feature_columns': [],
        'errors': [],
    }

    # ── 1. Load ────────────────────────────────────────────────────────────────
    df = pd.read_csv(file_path)
    summary['original_shape'] = list(df.shape)
    summary['steps'].append({
        'title': 'Dataset Loaded',
        'icon': 'cloud-check',
        'color': 'primary',
        'detail': f'{df.shape[0]:,} rows × {df.shape[1]} columns loaded from CSV.',
        'status': 'success',
    })

    # ── 2. Drop excluded columns ───────────────────────────────────────────────
    cols_to_drop = [c for c in excluded_columns if c in df.columns and c != target_column]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)
        summary['steps'].append({
            'title': 'Excluded Columns Dropped',
            'icon': 'dash-circle',
            'color': 'secondary',
            'detail': f'Dropped {len(cols_to_drop)} excluded column(s): {", ".join(cols_to_drop)}.',
            'status': 'info',
        })

    # ── 3. Remove duplicates ───────────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    dups = before - len(df)
    summary['duplicates_removed'] = dups
    summary['steps'].append({
        'title': 'Duplicate Rows Removed',
        'icon': 'files',
        'color': 'warning' if dups > 0 else 'success',
        'detail': f'{dups} duplicate row(s) found and removed.' if dups else 'No duplicate rows found.',
        'status': 'warning' if dups > 0 else 'success',
    })

    # ── 3.5 Auto-clean string columns that are actually formatted numbers (like currency/percentage) ──
    cleaned_cols = []
    for col in df.columns:
        if col == target_column or col not in excluded_columns:
            if pd.api.types.is_string_dtype(df[col]):
                non_null = df[col].dropna()
                if not non_null.empty:
                    # Sample a few to check if it looks like currency/percentage/number
                    sample = non_null.head(100).astype(str).str.strip()
                    cleaned_sample = (
                        sample.str.replace('$', '', regex=False)
                              .str.replace('%', '', regex=False)
                              .str.replace(',', '', regex=False)
                              .str.strip()
                    )
                    try:
                        converted_sample = pd.to_numeric(cleaned_sample, errors='coerce')
                        success_rate = converted_sample.notna().sum() / len(sample)
                        if success_rate > 0.8:
                            df[col] = pd.to_numeric(
                                df[col].astype(str)
                                       .str.replace('$', '', regex=False)
                                       .str.replace('%', '', regex=False)
                                       .str.replace(',', '', regex=False)
                                       .str.strip(),
                                errors='coerce'
                            )
                            cleaned_cols.append(col)
                    except Exception:
                        pass
    if cleaned_cols:
        summary['steps'].append({
            'title': 'Formatted Numeric Data Cleaned',
            'icon': 'magic',
            'color': 'info',
            'detail': f'Auto-cleaned and converted {len(cleaned_cols)} string column(s) to numeric: {", ".join(cleaned_cols)}.',
            'status': 'success',
        })

    # ── 4. Validate target ─────────────────────────────────────────────────────
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found after dropping excluded columns.")

    # ── 5. Separate features / target ─────────────────────────────────────────
    X = df.drop(columns=[target_column])
    y = df[target_column]

    feature_columns = list(X.columns)
    summary['feature_columns'] = feature_columns
    summary['problem_type'] = detect_problem_type(y)
    summary['steps'].append({
        'title': 'Problem Type Detected',
        'icon': 'diagram-3',
        'color': 'primary',
        'detail': f'Target column detected as {summary["problem_type"].title()}.',
        'status': 'success',
    })

    # ── 6. Missing value handling ──────────────────────────────────────────────
    missing_log = {}
    for col in X.columns:
        n_missing = X[col].isnull().sum()
        if n_missing == 0:
            continue

        pct = round(n_missing / len(X) * 100, 1)

        if missing_strategy == 'drop' or (missing_strategy == 'auto' and pct > 50):
            X = X.drop(columns=[col])
            missing_log[col] = f'DROPPED ({pct}% missing)'
        elif pd.api.types.is_numeric_dtype(X[col]):
            fill_val = X[col].median()
            X[col] = X[col].fillna(fill_val)
            missing_log[col] = f'Filled with median ({fill_val:.4g}) — {n_missing} cells'
        else:
            fill_val = X[col].mode()[0] if not X[col].mode().empty else 'Unknown'
            X[col] = X[col].fillna(fill_val)
            missing_log[col] = f'Filled with mode ("{fill_val}") — {n_missing} cells'

    # Handle missing in target
    n_target_missing = y.isnull().sum()
    if n_target_missing > 0:
        X = X[y.notna()]
        y = y[y.notna()]
        missing_log[target_column] = f'Rows with missing target dropped — {n_target_missing} rows removed'

    summary['missing_handled'] = missing_log
    summary['steps'].append({
        'title': 'Missing Values Handled',
        'icon': 'bandaid',
        'color': 'info',
        'detail': (
            f'{len(missing_log)} column(s) had missing values. '
            'Numeric columns filled with median, categorical with mode, columns >50% missing dropped.'
        ) if missing_log else 'No missing values found — dataset is complete.',
        'status': 'info' if missing_log else 'success',
        'breakdown': missing_log,
    })

    # ── 7. Encode categorical columns ─────────────────────────────────────────
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    encoded_log = {}
    encoders = {}
    dropped_cols = []

    for col in categorical_cols:
        n_unique = X[col].nunique()
        if n_unique > 15:
            # High cardinality categorical feature (e.g. Name, Ticket, Cabin)
            X = X.drop(columns=[col])
            dropped_cols.append(col)
            encoded_log[col] = {
                'method': 'Dropped',
                'detail': f'Dropped high-cardinality category column ({n_unique} unique values).',
            }
        elif n_unique <= 2:
            # Binary → Label Encode
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
            encoded_log[col] = {
                'method': 'Label Encoding',
                'classes': list(le.classes_),
                'new_cols': [col],
            }
        else:
            # Multi-class → One-Hot Encode
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=False, dtype=int)
            X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
            encoded_log[col] = {
                'method': 'One-Hot Encoding',
                'new_cols': list(dummies.columns),
                'n_categories': n_unique,
            }

    # Encode target if categorical
    target_encoder = None
    target_classes = None
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), name=target_column)
        target_classes = list(target_encoder.classes_)
        encoded_log[target_column] = {
            'method': 'Label Encoding (target)',
            'classes': target_classes,
            'new_cols': [target_column],
        }

    summary['encoded_columns'] = encoded_log
    encoded_count = len(categorical_cols) - len(dropped_cols)
    detail_msg = f'{encoded_count} categorical column(s) encoded.'
    if dropped_cols:
        detail_msg += f' Dropped {len(dropped_cols)} high-cardinality column(s): {", ".join(dropped_cols)}.'
    summary['steps'].append({
        'title': 'Categorical Encoding',
        'icon': 'tag',
        'color': 'purple',
        'detail': detail_msg if categorical_cols else 'No categorical columns found — all features are numeric.',
        'status': 'info' if categorical_cols else 'success',
        'breakdown': encoded_log,
    })

    # ── 8. Scale numeric features ──────────────────────────────────────────────
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = None
    if numeric_cols and scale_method != 'none':
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    summary['scaled_columns'] = numeric_cols
    summary['steps'].append({
        'title': f'Numerical Scaling ({scale_method.title()})',
        'icon': 'rulers',
        'color': 'cyan',
        'detail': (
            f'{len(numeric_cols)} numeric feature(s) scaled using StandardScaler '
            '(mean=0, std=1).'
        ) if numeric_cols else 'No numeric columns to scale.',
        'status': 'info' if numeric_cols else 'success',
        'scaled': numeric_cols,
    })

    # ── 9. Train-test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    summary['train_samples'] = int(len(X_train))
    summary['test_samples']  = int(len(X_test))
    summary['final_shape']   = list(X.shape)
    summary['feature_columns'] = list(X.columns)

    summary['steps'].append({
        'title': 'Train-Test Split',
        'icon': 'scissors',
        'color': 'success',
        'detail': (
            f'Dataset split: {int((1-test_size)*100)}% train / {int(test_size*100)}% test · '
            f'{len(X_train):,} training samples · {len(X_test):,} test samples.'
        ),
        'status': 'success',
    })

    # ── 10. Train models and collect scores ──────────────────────────────────
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    best_result = select_best_model(X_train, y_train, summary['problem_type'],
                                    save_dir=models_dir,
                                    X_test=X_test, y_test=y_test)
    summary['model_scores'] = {
        name: metrics['accuracy' if summary['problem_type'] != 'regression' else 'r2']
        for name, metrics in best_result['metrics'].items()
    }
    summary['best_model'] = {
        'name': best_result['best_model_name'],
        'path': best_result['best_model_path'],
        'metrics': best_result['metrics'][best_result['best_model_name']],
    }
    summary['visualizations'] = generate_visualizations(pd.concat([X_train, y_train], axis=1), y_train, summary, os.path.join(os.path.dirname(__file__), 'reports'))
    summary['steps'].append({
        'title': 'Model Training Complete',
        'icon': 'cpu',
        'color': 'success',
        'detail': f'Trained {len(best_result["metrics"])} {summary["problem_type"]} model(s) and selected "{best_result["best_model_name"]}" as the best performer.',
        'status': 'success',
    })

    # ── 11. Save artifacts ─────────────────────────────────────────────────────
    artifacts = {}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        def _save(obj, name):
            path = os.path.join(save_dir, name)
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
            return path

        artifacts['x_train_path']  = _save(X_train,  'X_train.pkl')
        artifacts['x_test_path']   = _save(X_test,   'X_test.pkl')
        artifacts['y_train_path']  = _save(y_train,  'y_train.pkl')
        artifacts['y_test_path']   = _save(y_test,   'y_test.pkl')
        artifacts['scaler_path']   = _save(scaler,   'scaler.pkl')   if scaler  else None
        artifacts['training_result_path'] = _save(best_result, 'training_result.pkl')
        artifacts['encoder_path']  = _save({'feature_encoders': encoders,
                                            'target_encoder': target_encoder,
                                            'target_classes': target_classes},
                                           'encoders.pkl')

    return {
        'summary':   summary,
        'artifacts': artifacts,
        'X_train':   X_train,
        'X_test':    X_test,
        'y_train':   y_train,
        'y_test':    y_test,
    }

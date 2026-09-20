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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')


def read_csv_file(file_path):
    """Read common CSV exports with detected delimiter and encoding fallback."""
    for encoding in ('utf-8-sig', 'latin1'):
        try:
            return pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    raise ValueError('Unable to read the CSV file with supported encodings.')


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


def clean_dataset(
    file_path: str,
    columns_to_drop: list = None,
    missing_strategy: str = 'auto',
    remove_duplicates: bool = True,
    fix_formatted_numbers: bool = True,
) -> dict:
    """
    Standalone data-cleaning pipeline (no ML, no train-test split).

    Parameters
    ----------
    file_path          : Path to the source CSV.
    columns_to_drop    : List of column names to remove before cleaning.
    missing_strategy   : One of 'auto', 'mean', 'median', 'mode', 'drop_rows'.
                         'auto' → numeric → median, categorical → mode,
                                  column dropped when >50 % missing.
    remove_duplicates  : Whether to remove exact duplicate rows.
    fix_formatted_numbers : Whether to strip $, %, commas from string columns
                            that are actually numeric.

    Returns
    -------
    dict with keys:
        df      – cleaned pandas DataFrame
        report  – list of step-log dicts compatible with the pipeline log UI
        original_shape  – (rows, cols) before cleaning
        cleaned_shape   – (rows, cols) after cleaning
    """
    columns_to_drop = columns_to_drop or []
    report = []

    # ── 1. Load ────────────────────────────────────────────────────────────────
    df = read_csv_file(file_path)
    original_shape = df.shape
    report.append({
        'title': 'Dataset Loaded',
        'icon': 'cloud-check',
        'color': 'primary',
        'detail': f'{df.shape[0]:,} rows × {df.shape[1]} columns read from CSV.',
        'status': 'success',
    })

    # ── 2. Drop user-selected columns ──────────────────────────────────────────
    valid_drops = [c for c in columns_to_drop if c in df.columns]
    if valid_drops:
        df.drop(columns=valid_drops, inplace=True)
        report.append({
            'title': 'Columns Dropped',
            'icon': 'dash-circle',
            'color': 'secondary',
            'detail': f'Dropped {len(valid_drops)} column(s): {", ".join(valid_drops)}.',
            'status': 'info',
        })

    # ── 3. Remove duplicates ───────────────────────────────────────────────────
    if remove_duplicates:
        before = len(df)
        df.drop_duplicates(inplace=True)
        dups = before - len(df)
        report.append({
            'title': 'Duplicate Rows Removed',
            'icon': 'files',
            'color': 'warning' if dups > 0 else 'success',
            'detail': f'{dups} duplicate row(s) removed.' if dups else 'No duplicate rows found.',
            'status': 'warning' if dups > 0 else 'success',
        })

    # ── 4. Auto-fix formatted numeric strings ($, %, commas) ──────────────────
    if fix_formatted_numbers:
        cleaned_cols = []
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                non_null = df[col].dropna()
                if non_null.empty:
                    continue
                sample = non_null.head(100).astype(str).str.strip()
                cleaned_sample = (
                    sample.str.replace('$', '', regex=False)
                          .str.replace('%', '', regex=False)
                          .str.replace(',', '', regex=False)
                          .str.strip()
                )
                try:
                    converted = pd.to_numeric(cleaned_sample, errors='coerce')
                    if converted.notna().sum() / len(sample) > 0.8:
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
            report.append({
                'title': 'Formatted Numbers Fixed',
                'icon': 'magic',
                'color': 'info',
                'detail': (
                    f'Stripped currency/percentage symbols from {len(cleaned_cols)} '
                    f'column(s): {", ".join(cleaned_cols)}.'
                ),
                'status': 'success',
            })
        else:
            report.append({
                'title': 'Formatted Numbers Check',
                'icon': 'magic',
                'color': 'success',
                'detail': 'No currency/percentage formatted columns detected.',
                'status': 'success',
            })

    # ── 5. Handle missing values (Highly Precise) ────────────────────────────────
    missing_log = {}

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    # 5a. Categorical columns imputation (using mode)
    for col in categorical_cols:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            if missing_strategy == 'drop_rows':
                df = df[df[col].notna()]
                missing_log[col] = f'Rows with missing values dropped — {n_missing} row(s) removed'
            else:
                fill_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(fill_val)
                missing_log[col] = f'Imputed using Mode ("{fill_val}") — {n_missing} cell(s)'

    # 5b. Numerical columns imputation (using KNNImputer for high precision, or mean/median/drop)
    cols_to_knn = []
    for col in numeric_cols:
        n_missing = int(df[col].isnull().sum())
        if n_missing > 0:
            if missing_strategy == 'drop_rows':
                df = df[df[col].notna()]
                missing_log[col] = f'Rows with missing values dropped — {n_missing} row(s) removed'
            elif missing_strategy == 'auto' and (n_missing / len(df) > 0.5):
                df.drop(columns=[col], inplace=True)
                missing_log[col] = f'COLUMN DROPPED (highly sparse: {round(n_missing/len(df)*100)}% missing)'
            elif missing_strategy == 'mean':
                fill_val = df[col].mean()
                df[col] = df[col].fillna(fill_val)
                missing_log[col] = f'Imputed using Mean ({fill_val:.4g}) — {n_missing} cell(s)'
            elif missing_strategy == 'median':
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                missing_log[col] = f'Imputed using Median ({fill_val:.4g}) — {n_missing} cell(s)'
            elif missing_strategy == 'mode':
                fill_val = df[col].mode()[0] if not df[col].mode().empty else 0
                df[col] = df[col].fillna(fill_val)
                missing_log[col] = f'Imputed using Mode ({fill_val}) — {n_missing} cell(s)'
            else:
                # strategy is 'auto' (KNN Imputer for precise, multi-variable imputation)
                cols_to_knn.append(col)

    if cols_to_knn:
        try:
            from sklearn.impute import KNNImputer
            from sklearn.preprocessing import StandardScaler

            # KNN Imputer requires scaling to prevent columns with larger scales from dominating the distance metric
            # We scale all numeric columns that we have, impute, then inverse transform
            numeric_df = df[numeric_cols].copy()
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_df)

            imputer = KNNImputer(n_neighbors=5)
            imputed_scaled = imputer.fit_transform(scaled_data)

            imputed_data = scaler.inverse_transform(imputed_scaled)
            imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)

            for col in cols_to_knn:
                n_missing = int(df[col].isnull().sum())
                df[col] = imputed_df[col]
                missing_log[col] = f'Imputed using KNN (k=5 neighbors) for high precision — {n_missing} cell(s)'
        except Exception as e:
            # Fallback to median if KNN fails
            for col in cols_to_knn:
                n_missing = int(df[col].isnull().sum())
                fill_val = df[col].median()
                df[col] = df[col].fillna(fill_val)
                missing_log[col] = f'Imputed using Median (KNN fallback: {e}) — {n_missing} cell(s)'

    if missing_log:
        report.append({
            'title': 'Missing Values Handled',
            'icon': 'bandaid',
            'color': 'info',
            'detail': f'{len(missing_log)} column(s) had missing values processed.',
            'status': 'info',
            'breakdown': missing_log,
        })
    else:
        report.append({
            'title': 'Missing Values Check',
            'icon': 'bandaid',
            'color': 'success',
            'detail': 'No missing values detected — dataset is complete.',
            'status': 'success',
        })

    cleaned_shape = df.shape
    report.append({
        'title': 'Cleaning Complete',
        'icon': 'check-circle',
        'color': 'success',
        'detail': (
            f'Dataset cleaned: {original_shape[0]:,} → {cleaned_shape[0]:,} rows, '
            f'{original_shape[1]} → {cleaned_shape[1]} columns.'
        ),
        'status': 'success',
    })

    return {
        'df': df,
        'report': report,
        'original_shape': list(original_shape),
        'cleaned_shape': list(cleaned_shape),
    }


def train_models(X: pd.DataFrame, y: pd.Series, problem_type: str,
                 X_test: pd.DataFrame = None, y_test: pd.Series = None,
                 tuning_mode: str = 'quick',
                 selected_models: list = None,
                 optimization_metric: str = None) -> dict:
    """Train a suite of models for classification or regression and return scores.

    Supports Quick Train vs Deep Hyperparameter Search (RandomizedSearchCV)
    and custom algorithm selection.
    """
    n_est = 50 if len(X) > 20000 else 100

    if problem_type == 'regression':
        available_models = {
            'linear_regression': LinearRegression(),
            'decision_tree_regressor': DecisionTreeRegressor(random_state=42),
            'random_forest_regressor': RandomForestRegressor(n_estimators=n_est, random_state=42),
            'gradient_boosting_regressor': GradientBoostingRegressor(n_estimators=n_est, random_state=42),
        }
        param_grids = {
            'decision_tree_regressor': {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
            },
            'random_forest_regressor': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
            },
            'gradient_boosting_regressor': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5],
            },
        }
    else:
        available_models = {
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=n_est, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(n_estimators=n_est, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
        }
        if len(X) <= 15000:
            available_models['svm'] = SVC(random_state=42, probability=True)

        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
            },
            'decision_tree': {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
            },
            'random_forest': {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
            },
            'gradient_boosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5],
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['rbf', 'linear'],
            },
        }

    # Filter to selected models if provided
    if selected_models:
        models = {k: v for k, v in available_models.items() if k in selected_models}
        if not models:
            models = available_models
    else:
        models = available_models

    X_eval = X_test if X_test is not None else X
    y_eval = y_test if y_test is not None else y

    trained_models = {}
    scores = {}
    best_params_log = {}

    for name, base_model in models.items():
        fitted_model = base_model
        if tuning_mode == 'deep' and name in param_grids and len(X) >= 15:
            try:
                grid = param_grids[name]
                scoring_func = 'r2' if problem_type == 'regression' else 'f1_weighted'
                search = RandomizedSearchCV(
                    base_model,
                    param_distributions=grid,
                    n_iter=min(5, sum(len(v) for v in grid.values())),
                    cv=min(3, max(2, len(X) // 10)),
                    scoring=scoring_func,
                    random_state=42,
                    n_jobs=-1,
                )
                search.fit(X, y)
                fitted_model = search.best_estimator_
                best_params_log[name] = search.best_params_
            except Exception:
                fitted_model.fit(X, y)
        else:
            fitted_model.fit(X, y)

        preds = fitted_model.predict(X_eval)
        if problem_type == 'regression':
            scores[name] = float(r2_score(y_eval, preds))
        else:
            scores[name] = float(accuracy_score(y_eval, preds))

        trained_models[name] = {
            'model': fitted_model,
            'predictions': preds,
            'y_eval': y_eval,
            'best_params': best_params_log.get(name, None),
        }

    return {
        'problem_type': problem_type,
        'trained_models': trained_models,
        'models': list(models.keys()),
        'scores': scores,
        'best_params_log': best_params_log,
        'tuning_mode': tuning_mode,
    }


def _save_chart(fig, save_path: str) -> str:
    """Save a matplotlib figure to disk and return the path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def generate_visualizations(df: pd.DataFrame, y: pd.Series, summary: dict, save_dir: str,
                            best_model=None) -> dict:
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

        # Box plot for numeric spread and outlier visibility.
        boxplot_df = numeric_df.iloc[:, :12]
        fig, ax = plt.subplots(figsize=(max(7, len(boxplot_df.columns) * 0.8), 5))
        sns.boxplot(data=boxplot_df, ax=ax, color='#14B8A6')
        ax.set_title('Numeric Feature Distributions')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        charts['numeric_boxplot'] = _save_chart(fig, os.path.join(save_dir, 'numeric_boxplot.png'))

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

    # Feature importance graph from the selected model.
    feature_names = [c for c in numeric_df.columns if c != y.name]
    importance = None
    if best_model is not None and feature_names:
        if hasattr(best_model, 'feature_importances_'):
            values = np.asarray(best_model.feature_importances_)
            if len(values) == len(feature_names):
                importance = pd.Series(values, index=feature_names)
        elif hasattr(best_model, 'coef_'):
            values = np.asarray(best_model.coef_)
            if values.ndim > 1:
                values = np.mean(np.abs(values), axis=0)
            else:
                values = np.abs(values)
            if len(values) == len(feature_names):
                importance = pd.Series(values, index=feature_names)
    if importance is not None and not importance.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        importance.nlargest(12).sort_values().plot(kind='barh', ax=ax, color='mediumseagreen')
        ax.set_title('Top Feature Importance')
        ax.set_xlabel('Importance')
        charts['feature_importance'] = _save_chart(fig, os.path.join(save_dir, 'feature_importance.png'))

        strongest_feature = importance.idxmax()
        if strongest_feature in df.columns and y.name in df.columns:
            relationship_df = df[[strongest_feature, y.name]].dropna()
            if not relationship_df.empty:
                fig, ax = plt.subplots(figsize=(7, 4))
                sns.scatterplot(data=relationship_df, x=strongest_feature, y=y.name, ax=ax,
                                color='#F59E0B', alpha=0.65)
                ax.set_title(f'{strongest_feature} vs. {y.name}')
                charts['strongest_feature_relationship'] = _save_chart(
                    fig, os.path.join(save_dir, 'strongest_feature_relationship.png')
                )

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
                      y_test: pd.Series = None,
                      tuning_mode: str = 'quick',
                      selected_models: list = None,
                      optimization_metric: str = None) -> dict:
    """Train models, compare metrics on the test split, select the best model based on chosen optimization metric."""
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), 'models')

    training_result = train_models(
        X, y, problem_type,
        X_test=X_test, y_test=y_test,
        tuning_mode=tuning_mode,
        selected_models=selected_models,
        optimization_metric=optimization_metric,
    )
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
        metric_key = optimization_metric if optimization_metric in {'r2', 'rmse', 'mae'} else 'r2'
        if metric_key == 'r2':
            best_name = max(metrics_by_model, key=lambda n: metrics_by_model[n]['r2'])
            sort_ascending = False
        else:
            best_name = min(metrics_by_model, key=lambda n: metrics_by_model[n][metric_key])
            sort_ascending = True
        best_metric = metric_key
    else:
        for name, info in training_result['trained_models'].items():
            preds  = info['predictions']
            y_eval = info['y_eval']
            metrics_by_model[name] = {
                'accuracy':  float(accuracy_score(y_eval, preds)),
                'precision': float(precision_score(y_eval, preds, average='weighted', zero_division=0)),
                'recall':    float(recall_score(y_eval, preds, average='weighted', zero_division=0)),
                'f1':        float(f1_score(y_eval, preds, average='weighted', zero_division=0)),
            }
        metric_key = optimization_metric if optimization_metric in {'accuracy', 'precision', 'recall', 'f1'} else 'f1'
        best_name = max(metrics_by_model, key=lambda n: (metrics_by_model[n][metric_key], metrics_by_model[n]['accuracy']))
        best_metric = metric_key
        sort_ascending = False

    best_model_info = training_result['trained_models'][best_name]
    best_model = best_model_info['model']
    best_model_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        best_model_path = os.path.join(save_dir, f'{best_name}.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(best_model, f)

    return {
        'problem_type': problem_type,
        'best_model_name': best_name,
        'best_model': best_model,
        'best_model_path': best_model_path,
        'metrics': metrics_by_model,
        'best_metric': best_metric,
        'sort_ascending': sort_ascending,
        'best_params': best_model_info.get('best_params'),
        'best_params_log': training_result.get('best_params_log', {}),
        'tuning_mode': tuning_mode,
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
    problem_type: str = None,
    tuning_mode: str = 'quick',
    selected_models: list = None,
    optimization_metric: str = None,
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
    df = read_csv_file(file_path)
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
    summary['problem_type'] = (
        problem_type if problem_type in {'classification', 'regression'}
        else detect_problem_type(y)
    )
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
    models_dir = save_dir or os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    best_result = select_best_model(
        X_train, y_train, summary['problem_type'],
        save_dir=models_dir,
        X_test=X_test, y_test=y_test,
        tuning_mode=tuning_mode,
        selected_models=selected_models,
        optimization_metric=optimization_metric,
    )
    summary['model_scores'] = {
        name: metrics.get(best_result.get('best_metric', 'accuracy'), list(metrics.values())[0])
        for name, metrics in best_result['metrics'].items()
    }
    summary['best_model'] = {
        'name': best_result['best_model_name'],
        'path': best_result['best_model_path'],
        'metrics': best_result['metrics'][best_result['best_model_name']],
        'best_params': best_result.get('best_params'),
    }
    summary['tuning_mode'] = tuning_mode
    summary['optimization_metric'] = best_result.get('best_metric', 'default')
    summary['all_metrics'] = best_result['metrics']

    reports_dir = save_dir or os.path.join(os.path.dirname(__file__), 'reports')
    summary['visualizations'] = generate_visualizations(
        pd.concat([X_train, y_train], axis=1), y_train, summary, reports_dir,
        best_model=best_result['best_model']
    )
    tuning_lbl = "Deep RandomizedSearchCV Search" if tuning_mode == 'deep' else "Quick Baseline Train"
    summary['steps'].append({
        'title': f'AutoML Model Training ({tuning_mode.title()} Mode)',
        'icon': 'cpu',
        'color': 'success',
        'detail': (
            f'Trained {len(best_result["metrics"])} model(s) using {tuning_lbl}. '
            f'Selected "{best_result["best_model_name"].replace("_", " ").title()}" as winning model '
            f'optimized for {best_result.get("best_metric", "performance").upper()}.'
        ),
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

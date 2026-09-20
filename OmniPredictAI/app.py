import os
import uuid
import json
import pickle
import re
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from difflib import SequenceMatcher
from functools import wraps
from datetime import timedelta
from io import BytesIO

from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import pandas as pd
import numpy as np
from flask import (Flask, render_template, redirect, url_for,
                   request, session, g, flash, jsonify, send_file, send_from_directory, abort)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

from database import db
import preprocessing as pp_engine

load_dotenv()

# ─── App Configuration ────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY') or os.urandom(32)
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', '').lower() in {'1', 'true', 'yes'}
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
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
    try:
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    except Exception:
        return None


def validate_dataset_file(file_path):
    """Validate uploaded CSV data and raise a clear error for unusable datasets."""
    try:
        df = read_csv_file(file_path)
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


def read_csv_file(file_path):
    """Read common CSV exports, including BOM, Latin-1, and non-comma delimiters."""
    last_error = None
    for encoding in ('utf-8-sig', 'latin1'):
        try:
            return pd.read_csv(file_path, encoding=encoding, sep=None, engine='python')
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_error = exc
    raise ValueError(f'Unable to read the CSV file: {last_error}')


def fix_visualization_paths(summary):
    """Normalize visualization chart file paths for templates and local routing."""
    if summary and 'visualizations' in summary:
        fixed_visuals = {}
        for key, filepath in summary['visualizations'].items():
            if filepath:
                filename = os.path.basename(filepath)
                fixed_visuals[key] = filename
        summary['visualizations'] = fixed_visuals
    return summary

def _normalise_search_text(value):
    return re.sub(r'[^a-z0-9]+', ' ', str(value).lower()).strip()


def find_relevant_dataset_rows(question, df, limit=20):
    """Find row-level matches across the complete dataframe, including minor typos and column header filtering."""
    lowered_question = _normalise_search_text(question)

    # Extract clean search text by checking phrase after 'of' or 'for'
    search_text = lowered_question
    after_of = search_text.split(' of ')
    if len(after_of) > 2:
        search_text = after_of[-1]
    else:
        phrase_match = re.search(r'\b(?:of|for|about)\s+(.+?)(?:\?|$)', search_text)
        if phrase_match:
            search_text = phrase_match.group(1)

    # Strip column header names and domain terms from search_text
    column_terms_to_strip = [
        'date of birth', 'birth date', 'dob', 'd o b', 'birth',
        'contact number', 'phone number', 'mobile number', 'contact no', 'contact', 'phone', 'mobile',
        'student number', 'student id', 'student no', 'roll number', 'roll no',
        'serial number', 'sr no', 'srno', 's no', 'sr.no', 'sr',
        'program name', 'program', 'course', 'branch',
        'blood group', 'blood type', 'blood',
        'validity', 'address', 'location', 'residence',
        'rating', 'score', 'comment', 'comments', 'review', 'reviews', 'feedback'
    ]
    for col in df.columns:
        column_terms_to_strip.append(_normalise_search_text(col))

    for term in sorted(column_terms_to_strip, key=len, reverse=True):
        if term:
            search_text = re.sub(rf'\b{re.escape(term)}\b', ' ', search_text)

    ignored_terms = {
        'what', 'which', 'who', 'where', 'when', 'how', 'many', 'is', 'are', 'the',
        'a', 'an', 'of', 'for', 'about', 'please', 'tell', 'me', 'give', 'show',
        'find', 'contact', 'number', 'phone', 'mobile', 'value', 'values', 'data',
        'i', 'want', 'get', 'show', 'please', 'provide', 'return',
        'record', 'row', 'rows', 'details', 'detail', 'information', 'info', 'full',
        'entire', 'whole', 'complete', 'date',
        'everything', 'all', 'does', 'have', 'has', 'with', 'and', 'in', 'on',
        'only', 'just', 'both', 'each', 'either', 'their', 'his', 'her', 'its',
        'was', 'were', 'or', 'to', 'from', 'by', 'this', 'that', 'these', 'those'
    }

    search_terms = [term for term in search_text.split() if term not in ignored_terms and len(term) > 1]

    if not search_terms:
        phrase_match = re.search(r'\b(?:of|for|about)\s+(.+?)(?:\?|$)', lowered_question)
        raw_text = phrase_match.group(1) if phrase_match else lowered_question
        search_terms = [term for term in raw_text.split() if term not in ignored_terms and len(term) > 1]

    if not search_terms:
        return []

    text_columns = df.select_dtypes(include=['object', 'string', 'category', 'number']).columns
    if not len(text_columns):
        return []

    matches_with_scores = []
    for row_index, row in df[text_columns].iterrows():
        row_terms = set()
        for value in row:
            if pd.notna(value):
                row_terms.update(_normalise_search_text(value).split())
        matched_terms = sum(
            any(term in candidate or SequenceMatcher(None, term, candidate).ratio() >= 0.82
                for candidate in row_terms)
            for term in search_terms
        )
        if matched_terms > 0:
            matches_with_scores.append((matched_terms, row_index))

    if not matches_with_scores:
        return []

    matches_with_scores.sort(key=lambda x: x[0], reverse=True)
    max_score = matches_with_scores[0][0]

    best_matches = [df.loc[index].to_dict() for score, index in matches_with_scores if score == max_score]
    return best_matches[:limit]


def _find_requested_columns(question, df):
    """Identify specific column names requested in the user question."""
    norm_q = _normalise_search_text(question)
    compact_q = re.sub(r'[^a-z0-9]', '', question.lower())

    requested = []

    # Direct match against column names
    for col in df.columns:
        norm_col = _normalise_search_text(col)
        compact_col = re.sub(r'[^a-z0-9]', '', str(col).lower())
        if (norm_col and norm_col in norm_q) or (len(compact_col) >= 2 and compact_col in compact_q):
            if col not in requested:
                requested.append(col)

    # Alias dictionary mapping common domain terms to columns
    aliases = {
        'dob': ['dob', 'd o b', 'date of birth', 'birth date', 'bday', 'birthday', 'birth'],
        'sr.no': ['sr no', 'srno', 's no', 'sr.no', 'serial number', 'sr'],
        'student number': ['student number', 'student id', 'student no', 'roll number', 'roll no'],
        'contact no': ['contact no', 'contact', 'phone', 'mobile', 'phone number', 'contact number', 'telephone'],
        'address': ['address', 'location', 'residence', 'street'],
        'blood group': ['blood group', 'blood', 'blood type'],
        'gender': ['gender', 'sex'],
        'name': ['name', 'student name', 'customer name', 'person name'],
        'program': ['program name', 'program', 'course', 'branch'],
        'validity': ['validity', 'valid till', 'valid'],
        'rating': ['rating', 'score', 'stars', 'star'],
        'comments': ['comments', 'comment', 'review', 'reviews', 'feedback'],
    }

    for field_key, keywords in aliases.items():
        if any(kw in norm_q or re.sub(r'[^a-z0-9]', '', kw) in compact_q for kw in keywords):
            for col in df.columns:
                norm_col = _normalise_search_text(col)
                compact_col = re.sub(r'[^a-z0-9]', '', str(col).lower())
                if any(kw in norm_col or re.sub(r'[^a-z0-9]', '', kw) in compact_col for kw in keywords):
                    if col not in requested:
                        requested.append(col)

    return requested


def answer_comment_review_question(question, df):
    """Answer questions related to reviews, comments, ratings, and sentiment analysis."""
    q_lower = question.lower()

    comment_keywords = [
        'comment', 'comments', 'review', 'reviews', 'feedback', 'rating', 'ratings',
        'good comments', 'bad comments', 'worst review', 'best review', 'ratio',
        'sentiment', 'opinion', 'opinions', 'positive', 'negative'
    ]
    if not any(kw in q_lower for kw in comment_keywords):
        return None

    comment_cols = [col for col in df.columns if any(k in str(col).lower() for k in ('comment', 'review', 'feedback', 'text', 'description'))]
    rating_cols = [col for col in df.columns if any(k in str(col).lower() for k in ('rating', 'score', 'stars', 'useful'))]
    name_cols = [col for col in df.columns if any(k in str(col).lower() for k in ('name', 'customer', 'user', 'author', 'title'))]

    if not comment_cols and not rating_cols:
        return None

    comment_col = comment_cols[0] if comment_cols else None
    rating_col = rating_cols[0] if rating_cols else None
    name_col = name_cols[0] if name_cols else (df.columns[0] if len(df.columns) > 0 else None)

    pos_words = {'good', 'great', 'excellent', 'awesome', 'amazing', 'love', 'best', 'positive', 'superb', 'fantastic', 'nice', 'satisfied', 'worth', 'killer', 'smooth', 'perfect', 'decent', 'impressed', 'powerful'}
    neg_words = {'bad', 'worst', 'terrible', 'horrible', 'poor', 'hate', 'negative', 'issue', 'problem', 'bug', 'defect', 'irritating', 'waste', 'slow', 'heating', 'disappointed', 'fail', 'useless', 'broken', 'cheap', 'ads', 'ad'}

    good_rows = []
    bad_rows = []
    row_sentiments = []

    for idx, row in df.iterrows():
        comment_text = str(row[comment_col]) if comment_col and pd.notna(row[comment_col]) else ''
        title_text = str(row['Review Title']) if 'Review Title' in df.columns and pd.notna(row['Review Title']) else ''
        full_text = (title_text + ' ' + comment_text).lower()

        rating_val = None
        if rating_col and pd.notna(row[rating_col]):
            r_str = str(row[rating_col])
            r_match = re.search(r'(\d+(?:\.\d+)?)', r_str)
            if r_match:
                rating_val = float(r_match.group(1))

        words = set(re.findall(r'\b[a-z]+\b', full_text))
        pos_score = len(words.intersection(pos_words))
        neg_score = len(words.intersection(neg_words))

        is_good = False
        is_bad = False

        if rating_val is not None:
            if rating_val >= 4.0 or (rating_val >= 3.5 and pos_score > neg_score):
                is_good = True
            elif rating_val <= 2.5 or (rating_val <= 3.0 and neg_score > pos_score):
                is_bad = True
        else:
            if pos_score > neg_score and pos_score >= 1:
                is_good = True
            elif neg_score > pos_score and neg_score >= 1:
                is_bad = True

        row_sentiments.append({
            'idx': idx,
            'row': row,
            'rating': rating_val,
            'pos_score': pos_score,
            'neg_score': neg_score,
            'is_good': is_good,
            'is_bad': is_bad
        })

        if is_good:
            good_rows.append(row)
        elif is_bad:
            bad_rows.append(row)

    # Intent 1: Ratio of good and bad comments
    if re.search(r'\b(?:ratio|proportion|breakdown|percentage|count|stats|sentiment)\b', q_lower) and ('good' in q_lower or 'bad' in q_lower or 'review' in q_lower or 'comment' in q_lower):
        total = len(df)
        g_count = len(good_rows)
        b_count = len(bad_rows)
        neu_count = max(0, total - g_count - b_count)
        ratio_str = f"{g_count / max(1, b_count):.2f} : 1" if b_count > 0 else f"{g_count} : 0"

        msg = [
            "Sentiment Breakdown of Comments & Reviews:\n",
            f"• Good (Positive) Reviews: {g_count} ({g_count/total*100:.1f}%)",
            f"• Bad (Negative) Reviews: {b_count} ({b_count/total*100:.1f}%)",
            f"• Neutral / Mixed Reviews: {neu_count} ({neu_count/total*100:.1f}%)",
            f"• Total Analyzed: {total} reviews",
            f"\n📊 Ratio of Good to Bad Comments: {ratio_str}"
        ]
        return {'kind': 'answer', 'message': '\n'.join(msg), 'changes': []}

    # Intent 2: Tell / show good comments
    if re.search(r'\b(?:good|positive|best|top|praise|favorable)\b', q_lower) and not re.search(r'\bworst\b', q_lower):
        if not good_rows:
            good_rows = [r['row'] for r in sorted(row_sentiments, key=lambda x: (x['rating'] or 0, x['pos_score']), reverse=True)[:5]]
        lines = [f"Top Positive / Good Comments ({len(good_rows)} positive reviews found):\n"]
        for i, row in enumerate(good_rows[:5], 1):
            name = row.get(name_col, f"Review #{i}")
            r_val = row.get(rating_col, '') if rating_col else ''
            title = row.get('Review Title', '')
            comm = str(row.get(comment_col, ''))[:200]
            lines.append(f"{i}. {name} ({r_val}): {title}\n   \"{comm}...\"\n")
        return {'kind': 'answer', 'message': '\n'.join(lines), 'changes': []}

    # Intent 3: Tell / show bad comments
    if re.search(r'\b(?:bad|negative|issues?|problems?|complaints?|critical)\b', q_lower) and not re.search(r'\b(?:ratio|good)\b', q_lower):
        if not bad_rows:
            bad_rows = [r['row'] for r in sorted(row_sentiments, key=lambda x: (x['rating'] or 5, -x['neg_score']))[:5]]
        lines = [f"Top Negative / Bad Comments ({len(bad_rows)} critical reviews found):\n"]
        for i, row in enumerate(bad_rows[:5], 1):
            name = row.get(name_col, f"Review #{i}")
            r_val = row.get(rating_col, '') if rating_col else ''
            title = row.get('Review Title', '')
            comm = str(row.get(comment_col, ''))[:200]
            lines.append(f"{i}. {name} ({r_val}): {title}\n   \"{comm}...\"\n")
        return {'kind': 'answer', 'message': '\n'.join(lines), 'changes': []}

    # Intent 4: Worst review
    if re.search(r'\b(?:worst|lowest|most negative)\b', q_lower):
        sorted_worst = sorted(row_sentiments, key=lambda x: (x['rating'] if x['rating'] is not None else 5, -x['neg_score']))
        if sorted_worst:
            worst = sorted_worst[0]['row']
            details = []
            for col in df.columns:
                if pd.notna(worst[col]):
                    val_str = str(worst[col])
                    if len(val_str) > 300:
                        val_str = val_str[:300] + '...'
                    details.append(f"• {col}: {val_str}")
            return {
                'kind': 'answer',
                'message': f"Worst Review Record (Lowest Rating / Most Critical Feedback):\n\n" + '\n'.join(details),
                'changes': []
            }

    # Intent 5: Best review
    if re.search(r'\b(?:best|highest|top review)\b', q_lower):
        sorted_best = sorted(row_sentiments, key=lambda x: (x['rating'] if x['rating'] is not None else 0, x['pos_score']), reverse=True)
        if sorted_best:
            best = sorted_best[0]['row']
            details = []
            for col in df.columns:
                if pd.notna(best[col]):
                    val_str = str(best[col])
                    if len(val_str) > 300:
                        val_str = val_str[:300] + '...'
                    details.append(f"• {col}: {val_str}")
            return {
                'kind': 'answer',
                'message': f"Best Review Record (Highest Rating / Most Positive Feedback):\n\n" + '\n'.join(details),
                'changes': []
            }

    return None


def _get_best_name_column(df):
    """Find the best representative entity name column from dataframe columns."""
    cols = list(df.columns)
    for c in cols:
        clow = str(c).strip().lower()
        if clow in {'name', 'student name', 'customer name', 'full name', 'person name', 'user name', 'username', 'title', 'review title', 'product name'}:
            return c
    for c in cols:
        clow = str(c).strip().lower()
        if 'name' in clow and not any(w in clow for w in ('program', 'file', 'dir', 'path', 'folder', 'class', 'school')):
            return c
    for c in cols:
        clow = str(c).strip().lower()
        if any(w in clow for w in ('name', 'title', 'product', 'item', 'customer', 'user', 'student')):
            return c
    return cols[0] if cols else None


def answer_row_lookup_question(question, df, history=None):
    """Answer row-specific and field-specific questions from every row accurately."""
    q_lower = question.lower()
    norm_q = _normalise_search_text(question)

    requested_cols = _find_requested_columns(question, df)
    label_col = _get_best_name_column(df)

    # Resolve "both" or multiple entities from question or history
    entities = []
    if re.search(r'\b(?:and|&|,)\b', q_lower) and not re.search(r'\b(?:between|range)\b', q_lower):
        raw_chunks = [c.strip() for c in re.split(r'\b(?:and|&|,)\b', q_lower) if c.strip()]
        for chunk in raw_chunks:
            clean_chunk = re.sub(r'\b(?:dob|sr\.?no|student number|contact|address|details|info|give|tell|me|show|only|just|of|both|for)\b', '', chunk).strip()
            if len(clean_chunk) > 2:
                entities.append(clean_chunk)

    if not entities and 'both' in q_lower:
        recent_entities = []
        if history:
            for item in history[::-1]:
                q_hist = item['question'] if isinstance(item, dict) or hasattr(item, '__getitem__') else str(item)
                matched_rows = find_relevant_dataset_rows(q_hist, df)
                if matched_rows:
                    for r in matched_rows:
                        name_val = str(r.get(label_col, '')).strip()
                        if name_val and name_val not in recent_entities:
                            recent_entities.append(name_val)
                if len(recent_entities) >= 2:
                    break
        if len(recent_entities) >= 2:
            entities = recent_entities[:2]

    if not entities:
        phrase_match = re.search(r'\b(?:of|for|about)\s+(.+?)(?:\?|$)', norm_q)
        target_name = phrase_match.group(1) if phrase_match else norm_q
        target_name = re.sub(r'\b(?:only|just|show|tell|give|me|the|details|info|record)\b', '', target_name).strip()
        if target_name:
            entities = [target_name]
        else:
            entities = [question]

    all_matched_results = []
    for ent in entities:
        rows = find_relevant_dataset_rows(ent, df)
        if rows:
            all_matched_results.append((ent, rows))

    if not all_matched_results:
        return None

    asks_for_details = bool(re.search(
        r'\b(?:details?|information|info|everything|all|entire|whole|full|complete|show me|summary)\b',
        norm_q,
    ))

    if requested_cols and not asks_for_details:
        if len(all_matched_results) == 1 and len(all_matched_results[0][1]) == 1:
            row = all_matched_results[0][1][0]
            col_vals = []
            for col in requested_cols:
                val = row.get(col)
                val_str = '(empty)' if pd.isna(val) else str(val)
                col_vals.append(f"{col}: {val_str}")
            return {
                'kind': 'answer',
                'message': ", ".join(col_vals),
                'changes': []
            }

        response_lines = []
        for ent, rows in all_matched_results:
            for row in rows[:3]:
                entity_label = str(row.get(label_col, ent)).strip()
                col_vals = []
                for col in requested_cols:
                    val = row.get(col)
                    val_str = '(empty)' if pd.isna(val) else str(val)
                    col_vals.append(f"{col}: {val_str}")

                response_lines.append(f"• {entity_label}: " + ", ".join(col_vals))

        if response_lines:
            return {
                'kind': 'answer',
                'message': '\n'.join(response_lines),
                'changes': []
            }

    if len(all_matched_results) == 1 and len(all_matched_results[0][1]) == 1:
        ent, rows = all_matched_results[0]
        row = rows[0]
        details = '\n'.join(
            f'- {column}: {("(empty)" if pd.isna(value) else value)}'
            for column, value in row.items()
        )
        return {
            'kind': 'answer',
            'message': f'Found matching record:\n{details}',
            'changes': [],
        }

    response_lines = []
    for ent, rows in all_matched_results:
        for row in rows[:3]:
            entity_label = str(row.get(label_col, ent)).strip()
            if requested_cols:
                vals = [f"{c}: {row.get(c)}" for c in requested_cols]
                response_lines.append(f"• {entity_label}: " + ", ".join(vals))
            else:
                response_lines.append(f"• {entity_label}")

    return {
        'kind': 'answer',
        'message': "Found matching records:\n" + "\n".join(response_lines),
        'changes': []
    }


def is_dataset_update_request(question):
    return bool(re.search(
        r'\b(?:change|update|edit|set|replace|correct|fix|modify|rename|remove|delete|clear)\b',
        question.lower(),
    ))


def get_numeric_series(df, column):
    """Try to convert a pandas column to numeric, supporting formatted string numbers."""
    if column not in df.columns:
        return None
    s = df[column]
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce')
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        s_non_null = s.dropna().astype(str).str.strip()
        if s_non_null.empty:
            return None
        s_clean = (
            s_non_null
             .str.replace('$', '', regex=False)
             .str.replace('%', '', regex=False)
             .str.replace(',', '', regex=False)
             .str.strip()
        )
        converted = pd.to_numeric(s_clean, errors='coerce')
        if not s_clean.empty and (converted.notna().sum() / len(s_non_null)) > 0.5:
            return pd.to_numeric(
                s.astype(str)
                 .str.strip()
                 .str.replace('$', '', regex=False)
                 .str.replace('%', '', regex=False)
                 .str.replace(',', '', regex=False)
                 .str.strip(),
                errors='coerce'
            )
    return None


def get_all_numeric_columns(df):
    """Return a dict of {column_name: cleaned_numeric_series} for all numeric columns."""
    num_cols = {}
    for column in df.columns:
        num_s = get_numeric_series(df, column)
        if num_s is not None:
            values = num_s.dropna()
            if not values.empty:
                num_cols[str(column)] = values
    return num_cols


def build_assistant_context(df, dataset_name, question=None):
    """Create useful context using full-data summaries and question-relevant rows."""
    numeric_summary = {}
    num_cols_dict = get_all_numeric_columns(df)
    for column, values in num_cols_dict.items():
        numeric_summary[str(column)] = {
            'count': int(values.count()),
            'average': round(float(values.mean()), 6),
            'median': round(float(values.median()), 6),
            'minimum': round(float(values.min()), 6),
            'maximum': round(float(values.max()), 6),
        }
    categorical_summary = {}
    for column in df.columns:
        if str(column) not in num_cols_dict:
            counts = df[column].fillna('(missing)').astype(str).value_counts()
            categorical_summary[str(column)] = {
                'unique_values': int(df[column].nunique(dropna=True)),
                'counts': {str(value): int(count) for value, count in counts.head(20).items()},
            }
    profile = {
        'dataset_name': dataset_name,
        'shape': list(df.shape),
        'full_dataset_summary': {
            'rows': int(len(df)),
            'missing_by_column': {str(column): int(df[column].isna().sum()) for column in df.columns},
            'numeric_columns': numeric_summary,
            'categorical_columns': categorical_summary,
        },
        'columns': [
            {
                'name': str(column),
                'dtype': str(df[column].dtype),
                'missing': int(df[column].isna().sum()),
                'unique': int(df[column].nunique(dropna=True)),
            }
            for column in df.columns
        ],
        'sample_rows': json.loads(df.head(10).to_json(orient='records', date_format='iso')),
    }
    if question:
        profile['question_relevant_rows'] = find_relevant_dataset_rows(question, df)
    return json.dumps(profile, default=str)


def answer_exact_numeric_question(question, df):
    """Answer simple numeric questions from the complete dataset locally."""
    lowered_question = question.lower()
    mention_match = re.search(
        r'\b(?:how many|number of)\b.*\b(?:about|mention|mentions|contain|contains)\b\s+["\']?([\w -]+)',
        lowered_question,
    )
    if mention_match:
        topic = mention_match.group(1).strip(' "\'?.!,')
        text_columns = df.select_dtypes(include=['object', 'string']).columns
        matching_rows = df[text_columns].astype(str).apply(
            lambda column: column.str.contains(re.escape(topic), case=False, na=False)
        ).any(axis=1)
        return {
            'kind': 'answer',
            'message': f'{int(matching_rows.sum())} of {len(df)} rows mention "{topic}" in the text fields.',
            'changes': [],
        }

    percentage_question = re.search(r'\b(?:percentage|percent|proportion|share)\b', lowered_question)
    gender_columns = [
        column for column in df.columns
        if re.search(r'\b(?:sex|gender)\b', str(column), re.IGNORECASE)
    ]
    if percentage_question and gender_columns and re.search(r'\b(?:male|female)\b', lowered_question):
        column = gender_columns[0]
        values = df[column].dropna().astype(str).str.strip().str.lower()
        total = len(values)
        if total:
            counts = {label: int((values == label).sum()) for label in ('male', 'female')}
            percentages = {label: count / total * 100 for label, count in counts.items()}
            return {
                'kind': 'answer',
                'message': (
                    f"Using all {total} non-missing values in {column}: "
                    f"Male: {counts['male']} ({percentages['male']:.2f}%), "
                    f"Female: {counts['female']} ({percentages['female']:.2f}%)."
                ),
                'changes': [],
            }

    threshold_match = re.search(r'\b(?:above|over|greater than|more than)\s+(-?\d+(?:\.\d+)?)', lowered_question)
    average_question = re.search(r'\b(?:average|mean|averages|means)\b', lowered_question)
    if not threshold_match and not average_question:
        return None

    num_cols_dict = get_all_numeric_columns(df)
    threshold = float(threshold_match.group(1)) if threshold_match else None
    numeric_columns = list(num_cols_dict.keys())

    aliases = {
        'ticket price': ('fare',),
        'ticket cost': ('fare',),
        'ticket fare': ('fare',),
        'price': ('price', 'fare', 'cost', 'amount'),
        'cost': ('cost', 'fare', 'price', 'amount'),
        'amount': ('amount', 'fare', 'price', 'cost'),
        'sugar': ('sugar', 'sugars'),
        'years': ('age',),
        'year': ('age',),
    }
    generic_terms = {'ticket', 'price', 'cost', 'amount', 'year', 'years'}
    mentioned_columns = [
        str(column) for column in df.columns
        if str(column).lower() not in generic_terms
        and re.search(rf'\b{re.escape(str(column).lower())}\b', lowered_question)
    ]

    # If general average question with NO specific column mentioned
    if average_question and not mentioned_columns and not threshold_match:
        if num_cols_dict:
            lines = [f"Numeric column averages across all {len(df):,} rows:"]
            for col, values in num_cols_dict.items():
                lines.append(f"- {col}: average = {values.mean():.2f}")
            return {
                'kind': 'answer',
                'message': '\n'.join(lines),
                'changes': [],
            }

    requested_columns = list(mentioned_columns)
    for phrase, names in aliases.items():
        if phrase in lowered_question:
            requested_columns.extend(
                str(column) for column in numeric_columns
                if any(name in str(column).lower() for name in names)
            )
    requested_columns.extend(str(column) for column in numeric_columns)
    requested_columns = list(dict.fromkeys(requested_columns))
    for column in requested_columns:
        matching = [actual for actual in df.columns if str(actual).lower() == column.lower()]
        if not matching:
            continue
        col_name = matching[0]
        values = get_numeric_series(df, col_name)
        if values is None or values.dropna().empty:
            continue
        values = values.dropna()
        if average_question:
            average = values.mean()
            return {
                'kind': 'answer',
                'message': f'The average {col_name} is {average:.2f}, calculated from all {len(df)} rows in the dataset.',
                'changes': [],
            }
        count = int((values > threshold).sum())
        return {
            'kind': 'answer',
            'message': f'{count} rows have {col_name} above {threshold:g}. This count uses the entire dataset ({len(df)} rows), not just the sample.',
            'changes': [],
        }
    return None


def answer_full_dataset_question(question, df):
    """Answer common dataset questions using every row in the uploaded file."""
    lowered_question = question.lower()
    row_count, column_count = df.shape
    num_cols_dict = get_all_numeric_columns(df)

    if re.search(r'\b(?:how many|number of|count of)\b.*\b(?:rows?|records?|entries?)\b', lowered_question):
        return {
            'kind': 'answer',
            'message': f'The complete dataset contains {row_count:,} rows.',
            'changes': [],
        }
    if re.search(r'\b(?:how many|number of|count of)\b.*\bcolumns?\b|\bhow many fields\b', lowered_question):
        return {
            'kind': 'answer',
            'message': f'The complete dataset contains {column_count:,} columns.',
            'changes': [],
        }
    if re.search(r'\b(?:what are|list|show|name)\b.*\bcolumns?\b|\bcolumn names?\b', lowered_question):
        return {
            'kind': 'answer',
            'message': 'Columns in the complete dataset: ' + ', '.join(str(column) for column in df.columns),
            'changes': [],
        }
    # Check for missing / empty / blank value questions
    missing_pattern = r'\b(?:missing|null|empty|blank|unfilled|lacking|without|no|don\'?t have|do not have|does not have|doesn\'?t have|not specified|not set|not present)\b'
    if re.search(missing_pattern, lowered_question):
        # First check if a specific column is mentioned in the query
        matching_cols = [col for col in df.columns if str(col).lower() in lowered_question]
        if not matching_cols:
            for col in df.columns:
                col_norm = re.sub(r'[^a-z0-9]', '', str(col).lower())
                q_norm = re.sub(r'[^a-z0-9]', '', lowered_question)
                if col_norm and col_norm in q_norm and len(col_norm) > 2:
                    matching_cols.append(col)

        if matching_cols:
            target_col = max(matching_cols, key=lambda c: len(str(c)))
            series = df[target_col]
            is_empty = series.isna() | series.astype(str).str.strip().str.lower().isin(['', 'nan', 'none', 'null', 'n/a', '(missing)', 'blank'])
            missing_count = int(is_empty.sum())
            pct = (missing_count / row_count * 100) if row_count > 0 else 0
            if missing_count == 0:
                msg = f"All {row_count:,} records in the dataset have '{target_col}' specified (0 missing or empty values)."
            else:
                msg = f"Out of {row_count:,} total records, {missing_count:,} ({pct:.1f}%) do not have '{target_col}' specified (missing/empty)."
            return {
                'kind': 'answer',
                'message': msg,
                'changes': [],
            }

        # Otherwise, return general missing value summary across all columns
        missing_counts = {}
        for col in df.columns:
            series = df[col]
            is_empty = series.isna() | series.astype(str).str.strip().str.lower().isin(['', 'nan', 'none', 'null', 'n/a', '(missing)', 'blank'])
            cnt = int(is_empty.sum())
            if cnt > 0:
                missing_counts[str(col)] = cnt

        if missing_counts:
            missing_text = ', '.join(f"{col}: {cnt:,} ({cnt / row_count * 100:.1f}%)" for col, cnt in missing_counts.items())
            msg = f"Across all {row_count:,} rows, missing/empty values are: {missing_text}."
        else:
            msg = f"There are no missing or empty values across all {row_count:,} rows."
        return {
            'kind': 'answer',
            'message': msg,
            'changes': [],
        }

    # Handle general numeric columns & averages request (e.g., quick prompt "Which columns are numeric and what are their averages?")
    if re.search(
        r'\b(?:which columns are numeric|numeric averages|numeric columns|averages? of (?:all )?(?:numeric )?columns|show (?:all )?(?:numeric )?averages)\b',
        lowered_question,
    ) or (
        re.search(r'\b(?:average|mean|averages|means)\b', lowered_question)
        and re.search(r'\b(?:numeric|all|columns|dataset)\b', lowered_question)
        and not any(str(c).lower() in lowered_question for c in df.columns)
    ):
        if num_cols_dict:
            lines = [f"Found {len(num_cols_dict)} numeric column(s) across all {row_count:,} rows:"]
            for col_name, values in num_cols_dict.items():
                lines.append(
                    f"- {col_name}: average = {values.mean():.2f} (min: {values.min():g}, max: {values.max():g}, median: {values.median():.2f})"
                )
            return {
                'kind': 'answer',
                'message': '\n'.join(lines),
                'changes': [],
            }
        else:
            return {
                'kind': 'answer',
                'message': f"No numeric columns detected across all {row_count:,} rows.",
                'changes': [],
            }

    mentioned_column = next(
        (column for column in df.columns if str(column).lower() in lowered_question),
        None,
    )
    if mentioned_column in num_cols_dict and re.search(
        r'\b(?:average|mean|median|minimum|min|maximum|max|range|statistics|stats)\b',
        lowered_question,
    ):
        values = num_cols_dict[mentioned_column]
        return {
            'kind': 'answer',
            'message': (
                f'Using all {len(values):,} non-empty values in {mentioned_column}: '
                f'minimum {values.min():g}, maximum {values.max():g}, '
                f'average {values.mean():.2f}, median {values.median():.2f}.'
            ),
            'changes': [],
        }

    if re.search(r'\b(?:unique|distinct)\b', lowered_question) and mentioned_column is not None:
        unique_count = int(df[mentioned_column].nunique(dropna=True))
        return {
            'kind': 'answer',
            'message': f'{mentioned_column} has {unique_count:,} distinct non-empty values across the complete dataset.',
            'changes': [],
        }

    if mentioned_column is not None and re.search(r'\b(?:distribution|breakdown|count|values|categories|types|how many)\b', lowered_question):
        counts = df[mentioned_column].fillna('(missing)').astype(str).value_counts().head(10)
        items = [f"{val}: {cnt:,}" for val, cnt in counts.items()]
        return {
            'kind': 'answer',
            'message': f"Value breakdown for '{mentioned_column}' across all {row_count:,} rows:\n" + "\n".join(f"- {item}" for item in items),
            'changes': []
        }

    return None


def is_dataset_explanation_request(question):
    return bool(re.search(
        r'\b(?:explain|describe|overview|understand|tell me about)\b.*\b(?:dataset|data|file)\b|'
        r'\bwhat is this dataset\b|\bgive me (?:a )?(?:full|complete|detailed) explanation\b',
        question.lower(),
    ))


def explain_dataset(df, dataset_name, preprocessing_summary=None):
    """Build a complete, deterministic explanation from the full dataframe."""
    row_count, column_count = df.shape
    missing = df.isna().sum()
    missing_columns = [(str(column), int(count)) for column, count in missing.items() if count]
    numeric_columns = list(df.select_dtypes(include='number').columns)
    categorical_columns = list(df.select_dtypes(include=['object', 'string', 'category', 'bool']).columns)
    duplicate_count = int(df.duplicated().sum())

    lines = [
        f'Dataset: {dataset_name}',
        f'Overall shape: {row_count:,} rows and {column_count:,} columns.',
        f'Column types: {len(numeric_columns)} numeric, {len(categorical_columns)} categorical or boolean.',
        f'Duplicate rows: {duplicate_count:,}.',
    ]

    if missing_columns:
        missing_text = ', '.join(
            f'{column} ({count:,}, {count / row_count * 100:.1f}%)'
            for column, count in missing_columns
        )
        lines.append(f'Missing values: {missing_text}.')
    else:
        lines.append('Missing values: none.')

    if numeric_columns:
        lines.append('Numeric columns:')
        for column in numeric_columns:
            values = pd.to_numeric(df[column], errors='coerce').dropna()
            if values.empty:
                lines.append(f'- {column}: no usable numeric values.')
                continue
            lines.append(
                f'- {column}: range {values.min():g} to {values.max():g}; '
                f'average {values.mean():.2f}; median {values.median():.2f}; '
                f'{values.nunique():,} distinct values.'
            )

    if categorical_columns:
        lines.append('Categorical and boolean columns:')
        for column in categorical_columns:
            values = df[column].fillna('(missing)').astype(str)
            top_values = values.value_counts().head(5)
            distribution = ', '.join(f'{value} ({count:,})' for value, count in top_values.items())
            lines.append(
                f'- {column}: {values.nunique():,} distinct values; most common: {distribution}.'
            )

    if preprocessing_summary:
        target = preprocessing_summary.get('target_column')
        problem_type = preprocessing_summary.get('problem_type')
        feature_columns = preprocessing_summary.get('feature_columns', [])
        best_model = preprocessing_summary.get('best_model', {}).get('name')
        if target:
            lines.append(f'Machine-learning setup: target column is {target}; problem type is {problem_type or "not specified"}.')
        if feature_columns:
            lines.append(f'Model inputs: {len(feature_columns)} feature columns.')
        if best_model:
            lines.append(f'Best trained model: {best_model.replace("_", " ")}.')
    else:
        lines.append('Machine-learning setup: this dataset has not been preprocessed yet.')

    lines.append('This explanation is calculated from all rows in the uploaded CSV, not only a preview sample.')
    return '\n'.join(lines)

def answer_general_question(question):
    """Provide intelligent conversational answers for general questions, greetings, AI/ML concepts, and data science queries."""
    q_lower = question.strip().lower()

    # Greetings & Introductions
    if re.search(r'\b(?:hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy|hola)\b', q_lower):
        return {
            'kind': 'answer',
            'message': (
                "Hello! 👋 I am your OmniPredict AI Co-Pilot.\n\n"
                "I am here to assist you as a conversational AI chatbot! You can ask me:\n"
                "• General AI & ML Questions: Overfitting, classification vs regression, model evaluation metrics, algorithms, and data science best practices.\n"
                "• Python & Data Science: Tips on data cleaning, pandas, missing value imputation, and feature engineering.\n"
                "• Dataset Analysis: Explore columns, dataset summary, numeric averages, missing value checks, or row lookups.\n"
                "• Dataset Editing: Propose updates to dataset cells for your confirmation.\n\n"
                "How can I help you today?"
            ),
            'changes': [],
        }

    if re.search(r'\b(?:who are you|what is your name|what can you do|your role|help me|what are your capabilities|features)\b', q_lower):
        return {
            'kind': 'answer',
            'message': (
                "I am OmniPredict AI Co-Pilot, your interactive AI assistant for machine learning and predictive analytics!\n\n"
                "You can talk to me like a normal chatbot:\n"
                "1. General Chat & AI Knowledge: Ask about ML algorithms (Random Forest, Logistic Regression, XGBoost), statistics, or python code.\n"
                "2. Dataset Insights: Ask questions such as 'Which columns are numeric?', 'How many rows are in the file?', or 'Show missing values'.\n"
                "3. Data Corrections: Ask me to update a cell (e.g. 'Set row 3 price to 199'), and I will build a preview for you to review and confirm."
            ),
            'changes': [],
        }

    if re.search(r'\b(?:how are you|how is it going|how do you do|what\'s up|sup)\b', q_lower):
        return {
            'kind': 'answer',
            'message': "I'm doing great and ready to assist you with machine learning, data science, or your dataset! What's on your mind?",
            'changes': [],
        }

    if re.search(r'\b(?:thank you|thanks|thank u|ty|appreciate it)\b', q_lower):
        return {
            'kind': 'answer',
            'message': "You're very welcome! Feel free to ask any other questions about AI, machine learning, or your dataset.",
            'changes': [],
        }

    if re.search(r'\b(?:bye|goodbye|see ya|cya)\b', q_lower):
        return {
            'kind': 'answer',
            'message': "Goodbye! Have a great day, and feel free to return whenever you need assistance with your data or predictions.",
            'changes': [],
        }

    # AI / ML / Data Science Concepts
    if re.search(r'\boverfitting\b|\bunderfitting\b', q_lower):
        return {
            'kind': 'answer',
            'message': (
                "Overfitting vs. Underfitting in Machine Learning:\n\n"
                "• Overfitting: The model learns training data too well (including noise and outliers), scoring high on training data but failing on new test data.\n"
                "  Fixes: Use cross-validation, gather more data, apply regularization (L1/L2), reduce tree depth, or drop noisy features.\n\n"
                "• Underfitting: The model is too simple to capture underlying patterns (e.g., linear model on non-linear data).\n"
                "  Fixes: Use a higher capacity model, add more features, or tune hyperparameters."
            ),
            'changes': [],
        }

    if re.search(r'\b(?:classification|regression)\b', q_lower) and re.search(r'\b(?:vs|difference|between|mean|distinguish)\b', q_lower):
        return {
            'kind': 'answer',
            'message': (
                "Classification vs. Regression:\n\n"
                "• Classification: Used when predicting discrete categories or class labels (e.g., Spam vs. Not Spam, Customer Churn: Yes/No, Image Tagging).\n"
                "• Regression: Used when predicting continuous numerical targets (e.g., House Price, Temperature, Stock Price).\n\n"
                "OmniPredict AI automatically detects your target column type and selects the best classification or regression algorithms!"
            ),
            'changes': [],
        }

    if re.search(r'\b(?:precision|recall|f1|accuracy|metrics|roc|auc|confusion matrix)\b', q_lower):
        return {
            'kind': 'answer',
            'message': (
                "Key Machine Learning Evaluation Metrics:\n\n"
                "• Accuracy: Overall fraction of correct predictions (TP + TN) / Total. Best for balanced datasets.\n"
                "• Precision: Proportion of positive identifications that were correct TP / (TP + FP). Essential when False Positives are costly (e.g., spam filter).\n"
                "• Recall: Proportion of actual positives identified correctly TP / (TP + FN). Essential when False Negatives are costly (e.g., medical diagnosis).\n"
                "• F1-Score: Harmonic mean of Precision & Recall 2 * (Precision * Recall) / (Precision + Recall). Best for imbalanced classes.\n"
                "• R² / RMSE / MAE: Used for regression models to evaluate numerical error."
            ),
            'changes': [],
        }

    if re.search(r'\b(?:random forest|decision tree|logistic regression|linear regression|xgboost|svm|gradient boosting|k-means)\b', q_lower):
        return {
            'kind': 'answer',
            'message': (
                "Popular Machine Learning Algorithms:\n\n"
                "• Random Forest: Ensemble of decision trees using bagging. Highly accurate, prevents overfitting, handles missing values & non-linear relationships well.\n"
                "• Logistic Regression: Linear classifier outputting class probabilities using a sigmoid function.\n"
                "• Linear Regression: Fits a linear equation to continuous numerical targets.\n"
                "• Gradient Boosting / XGBoost: Builds decision trees sequentially to iteratively reduce prediction errors.\n"
                "• Support Vector Machines (SVM): Finds optimal decision hyperplanes maximizing margins."
            ),
            'changes': [],
        }

    if re.search(r'\b(?:missing value|impute|imputation|null|nan)\b', q_lower) and not re.search(r'\b(?:dataset|csv|this|column|here|rows)\b', q_lower):
        return {
            'kind': 'answer',
            'message': (
                "Best Practices for Handling Missing Data:\n\n"
                "1. Numerical Columns: Impute missing values using column Mean (if normally distributed) or Median (if skewed by outliers).\n"
                "2. Categorical Columns: Impute using the Mode (most frequent category) or tag as 'Unknown'.\n"
                "3. Removal: Drop rows or columns only if the missing proportion is very small (<5%) or very high (>60%).\n\n"
                "OmniPredict AI automatically handles missing value imputation during automated preprocessing!"
            ),
            'changes': [],
        }

    if re.search(r'\b(?:scaling|normalization|standardization)\b', q_lower):
        return {
            'kind': 'answer',
            'message': (
                "Feature Scaling: Normalization vs Standardization:\n\n"
                "• Standardization (StandardScaler): Rescales data to have a mean of 0 and standard deviation of 1 (x - mean) / std. Best for distance-based algorithms (SVM, KNN, Logistic Regression).\n"
                "• Normalization (MinMaxScaler): Rescales values into a range of [0, 1] (x - min) / (max - min). Best when features have non-Gaussian distributions."
            ),
            'changes': [],
        }

    return None


def request_dataset_assistant(question, context, df=None, dataset_name=None, preprocessing_summary=None):
    prompt = (
        'You are OmniPredict AI Co-Pilot, a friendly and intelligent AI assistant and dataset copilot. '
        'You can answer general questions, greetings, AI/ML concepts, data science advice, programming, '
        'and general conversational queries like a normal chatbot. '
        'Do not use double asterisks (**) for bolding in your responses. '
        'When the question is specifically about the uploaded dataset or requests dataset edits, '
        'use the supplied dataset context.\n\n'
        'Return JSON with exact top-level keys:\n'
        '- kind: "answer" (for general answers or dataset answers) or "update_preview" (for cell edit requests)\n'
        '- message: string containing your helpful, well-formatted response\n'
        '- changes: array of object [{row_number, column, old_value, new_value}] if requesting cell edit, else []\n\n'
        f'Dataset context:\n{context}\n\n'
        f'User question:\n{question}'
    )
    payload = json.dumps({
        'model': os.environ.get('OLLAMA_MODEL', 'llama3.2:3b'),
        'system': (
            'You are OmniPredict AI Co-Pilot, a friendly, intelligent AI chatbot assistant. '
            'Do not format text with double asterisks (**). '
            'Return only a JSON object with keys: kind, message, changes. '
            'kind must be "answer" or "update_preview". changes must be an array.'
        ),
        'prompt': prompt,
        'stream': False,
        'format': 'json',
        'options': {'temperature': 0.3},
    }).encode('utf-8')
    request = Request(
        os.environ.get('OLLAMA_URL', 'http://127.0.0.1:11434/api/generate'),
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    try:
        with urlopen(request, timeout=10) as response:
            result = json.loads(response.read().decode('utf-8'))
        parsed = json.loads(result['response'])
        if isinstance(parsed, dict) and parsed.get('kind') in {'answer', 'update_preview'}:
            parsed.setdefault('message', '')
            parsed.setdefault('changes', [])
            return parsed
    except Exception:
        pass

    # Check for smart conversational general question fallback when LLM service is unavailable
    general_ans = answer_general_question(question)
    if general_ans is not None:
        return general_ans

    # Deterministic fallback when LLM service is unavailable
    if df is not None:
        lowered = question.lower()
        if is_dataset_explanation_request(question) or any(w in lowered for w in ['dataset', 'data', 'csv', 'column', 'row', 'stat', 'summary', 'missing', 'pattern']):
            return {
                'kind': 'answer',
                'message': explain_dataset(df, dataset_name or 'Uploaded Dataset', preprocessing_summary),
                'changes': [],
            }
        else:
            return {
                'kind': 'answer',
                'message': (
                    f"I received your question: \"{question}\".\n\n"
                    "I am your OmniPredict AI Co-Pilot! You can ask me general questions about AI/ML, data science concepts, "
                    "programming, or specific questions about your uploaded dataset."
                ),
                'changes': [],
            }

    return {
        'kind': 'answer',
        'message': f'Analysis for "{question}": Context prepared. Uploaded dataset is ready.',
        'changes': [],
    }



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
    latest_dataset_id = None
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
            latest_dataset_id = dataset['id']
            latest_summary = json.loads(pre['summary_json']) if pre['summary_json'] else None
            if latest_summary:
                latest_summary = fix_visualization_paths(latest_summary)
            break

    return render_template(
        'dashboard.html',
        page_name='dashboard',
        datasets=datasets,
        latest_summary=latest_summary,
        latest_dataset_id=latest_dataset_id,
        latest_result=latest_result,
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
        assistant_history = db.get_assistant_history(g.user['id'], dataset_id),
    )


@app.route('/dataset/<int:dataset_id>/row/<int:row_number>')
@login_required
def dataset_row_details(dataset_id, row_number):
    """Return one complete CSV row using its 1-based data-row number."""
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        return jsonify({'error': 'Dataset not found.'}), 404
    try:
        df = validate_dataset_file(record['file_path'])
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400
    if row_number < 1 or row_number > len(df):
        return jsonify({'error': f'Row number must be between 1 and {len(df)}.'}), 400

    row = json.loads(
        pd.DataFrame([df.iloc[row_number - 1]])
        .to_json(orient='records', date_format='iso')
    )[0]
    return jsonify({'row_number': row_number, 'values': row})


@app.route('/dataset/<int:dataset_id>/row-search')
@login_required
def dataset_row_search(dataset_id):
    """Find complete rows by number or by text such as a person's name."""
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        return jsonify({'error': 'Dataset not found.'}), 404
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Enter a row number or search text.'}), 400
    try:
        df = validate_dataset_file(record['file_path'])
        if query.isdigit():
            row_number = int(query)
            if row_number < 1 or row_number > len(df):
                return jsonify({'error': f'Row number must be between 1 and {len(df)}.'}), 400
            matches = [(row_number, df.iloc[row_number - 1].to_dict())]
        else:
            rows = find_relevant_dataset_rows(query, df, limit=5)
            matches = []
            for row in rows:
                matching_indexes = df.index[
                    df.astype(str).eq(pd.Series(row, index=df.columns).astype(str)).all(axis=1)
                ].tolist()
                if matching_indexes:
                    matches.append((int(matching_indexes[0]) + 1, row))
        if not matches:
            return jsonify({'error': f'No matching row found for "{query}".'}), 404
        return jsonify({
            'matches': [
                {'row_number': row_number, 'values': json.loads(
                    pd.DataFrame([row]).to_json(orient='records', date_format='iso')
                )[0]}
                for row_number, row in matches
            ]
        })
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400

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
        num_s = get_numeric_series(df, col)
        stat = {
            'dtype':   dtypes[col],
            'missing': int(s.isnull().sum()),
            'missing_pct': round(s.isnull().sum() / len(df) * 100, 1) if len(df) else 0,
            'unique':  int(s.nunique()),
        }
        if num_s is not None and not num_s.empty:
            stat.update({
                'is_numeric': True,
                'mean':  round(float(num_s.mean()), 4),
                'std':   round(float(num_s.std()),  4) if len(num_s) > 1 else 0.0,
                'min':   round(float(num_s.min()),  4),
                'max':   round(float(num_s.max()),  4),
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
        try:
            test_size = float(request.form.get('test_size', 0.2))
        except (TypeError, ValueError):
            flash('Test size must be a number between 0.1 and 0.4.', 'danger')
            return redirect(url_for('preprocess', dataset_id=dataset_id))

        scale_method        = request.form.get('scale_method', 'standard')
        missing_strategy    = request.form.get('missing_strategy', 'auto')
        tuning_mode         = request.form.get('tuning_mode', 'quick')
        selected_models     = request.form.getlist('selected_models')
        optimization_metric = request.form.get('optimization_metric', 'auto')

        if not 0.1 <= test_size <= 0.4:
            flash('Test size must be between 0.1 and 0.4.', 'danger')
            return redirect(url_for('preprocess', dataset_id=dataset_id))
        if scale_method not in {'standard', 'none'}:
            flash('Invalid scaling method.', 'danger')
            return redirect(url_for('preprocess', dataset_id=dataset_id))
        if missing_strategy not in {'auto', 'median', 'drop'}:
            flash('Invalid missing-value strategy.', 'danger')
            return redirect(url_for('preprocess', dataset_id=dataset_id))
        if tuning_mode not in {'quick', 'deep'}:
            tuning_mode = 'quick'

        # Save artifacts per-dataset in uploads/<dataset_id>/
        save_dir = os.path.join(UPLOAD_FOLDER, f'dataset_{dataset_id}')

        try:
            result = pp_engine.run_preprocessing(
                file_path           = record['file_path'],
                target_column       = config['target_column'],
                excluded_columns    = excluded,
                problem_type        = config['problem_type'],
                test_size           = test_size,
                scale_method        = scale_method,
                missing_strategy    = missing_strategy,
                tuning_mode         = tuning_mode,
                selected_models     = selected_models if selected_models else None,
                optimization_metric = optimization_metric if optimization_metric != 'auto' else None,
                save_dir            = save_dir,
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

# ─── Data Cleaning ────────────────────────────────────────────────────────────

@app.route('/dataset/<int:dataset_id>/clean', methods=['GET'])
@login_required
def clean_dataset_page(dataset_id):
    """Render the interactive data-cleaning options page."""
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

    columns = list(df.columns)
    dtypes  = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = df.isnull().sum().to_dict()
    missing_pct = {
        col: round(cnt / len(df) * 100, 1) if len(df) else 0
        for col, cnt in missing.items()
    }
    num_rows, num_cols = df.shape
    total_missing = int(sum(missing.values()))
    duplicate_count = int(df.duplicated().sum())

    # Detect formatted numeric columns (e.g., columns containing currency, percentages, or commas)
    formatted_cols = []
    for col in columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            non_null = df[col].dropna()
            if not non_null.empty:
                sample = non_null.head(100).astype(str).str.strip()
                cleaned_sample = (
                    sample.str.replace('$', '', regex=False)
                          .str.replace('%', '', regex=False)
                          .str.replace(',', '', regex=False)
                          .str.strip()
                )
                try:
                    converted = pd.to_numeric(cleaned_sample, errors='coerce')
                    if len(sample) > 0 and (converted.notna().sum() / len(sample)) > 0.8:
                        formatted_cols.append(col)
                except Exception:
                    pass

    return render_template(
        'clean_dataset.html',
        page_name       = 'dashboard',
        record          = dict(record),
        columns         = columns,
        dtypes          = dtypes,
        missing         = missing,
        missing_pct     = missing_pct,
        num_rows        = num_rows,
        num_cols        = num_cols,
        total_missing   = total_missing,
        duplicate_count = duplicate_count,
        formatted_cols  = formatted_cols,
        size_display    = format_size(record['file_size'] or 0),
    )



@app.route('/dataset/<int:dataset_id>/clean/run', methods=['POST'])
@login_required
def run_clean_dataset(dataset_id):
    """Run the auto-cleaning routine and save results to disk."""
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dashboard'))

    # Collect form params
    columns_to_drop       = request.form.getlist('columns_to_drop')
    missing_strategy      = request.form.get('missing_strategy', 'auto')
    remove_duplicates     = request.form.get('remove_duplicates') == 'on'
    fix_formatted_numbers = request.form.get('fix_formatted_numbers') == 'on'

    save_dir = os.path.join(UPLOAD_FOLDER, f'dataset_{dataset_id}')
    os.makedirs(save_dir, exist_ok=True)

    try:
        result = pp_engine.clean_dataset(
            file_path             = record['file_path'],
            columns_to_drop       = columns_to_drop,
            missing_strategy      = missing_strategy,
            remove_duplicates     = remove_duplicates,
            fix_formatted_numbers = fix_formatted_numbers,
        )
    except Exception as e:
        flash(f'Cleaning failed: {e}', 'danger')
        return redirect(url_for('clean_dataset_page', dataset_id=dataset_id))

    # Save cleaned CSV file
    cleaned_path = os.path.join(save_dir, 'cleaned.csv')
    result['df'].to_csv(cleaned_path, index=False)

    # Save summary report metadata
    summary = {
        'original_shape': result['original_shape'],
        'cleaned_shape': result['cleaned_shape'],
        'report': result['report'],
        'cleaned_at': g.user['full_name']
    }
    with open(os.path.join(save_dir, 'clean_report.json'), 'w') as f:
        json.dump(summary, f)

    flash('Dataset cleaned successfully! Review the changes below.', 'success')
    return redirect(url_for('clean_results_page', dataset_id=dataset_id))


@app.route('/dataset/<int:dataset_id>/clean/results', methods=['GET'])
@login_required
def clean_results_page(dataset_id):
    """Show audit logs and before/after details of the cleaned dataset."""
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dashboard'))

    report_path = os.path.join(UPLOAD_FOLDER, f'dataset_{dataset_id}', 'clean_report.json')
    if not os.path.exists(report_path):
        flash('No cleaning history found for this dataset.', 'warning')
        return redirect(url_for('clean_dataset_page', dataset_id=dataset_id))

    with open(report_path, 'r') as f:
        summary = json.load(f)

    # Get preview of cleaned data
    cleaned_csv_path = os.path.join(UPLOAD_FOLDER, f'dataset_{dataset_id}', 'cleaned.csv')
    preview_rows = []
    columns = []
    if os.path.exists(cleaned_csv_path):
        try:
            df_cleaned = read_csv_file(cleaned_csv_path)
            preview_rows = df_cleaned.head(10).to_dict(orient='records')
            columns = list(df_cleaned.columns)
        except Exception:
            pass

    return render_template(
        'clean_results.html',
        page_name    = 'dashboard',
        record       = dict(record),
        summary      = summary,
        preview_rows = preview_rows,
        columns      = columns,
        size_display = format_size(record['file_size'] or 0),
    )


@app.route('/dataset/<int:dataset_id>/clean/download', methods=['GET'])
@login_required
def download_clean_dataset(dataset_id):
    """Stream the previously cleaned CSV file to the browser."""
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dashboard'))

    cleaned_path = os.path.join(UPLOAD_FOLDER, f'dataset_{dataset_id}', 'cleaned.csv')
    if not os.path.exists(cleaned_path):
        flash('Cleaned file not found. Please run auto-clean first.', 'danger')
        return redirect(url_for('clean_dataset_page', dataset_id=dataset_id))

    base_name = record['original_name'].rsplit('.', 1)[0]
    download_name = f"{base_name}_cleaned.csv"

    return send_file(
        cleaned_path,
        mimetype='text/csv',
        as_attachment=True,
        download_name=download_name,
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

@app.route('/report-images/<int:dataset_id>/<path:filename>')
@login_required
def serve_report_image(dataset_id, filename):
    if not db.get_dataset_by_id(dataset_id, g.user['id']):
        abort(404)
    reports_dir = os.path.join(UPLOAD_FOLDER, f'dataset_{dataset_id}')
    scoped_path = os.path.join(reports_dir, filename)
    if os.path.isfile(scoped_path):
        return send_from_directory(reports_dir, filename)

    # Keep reports created before dataset-scoped artifacts were introduced visible.
    legacy_reports_dir = os.path.join(os.path.dirname(__file__), 'reports')
    if os.path.isfile(os.path.join(legacy_reports_dir, filename)):
        return send_from_directory(legacy_reports_dir, filename)

    abort(404)

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
                'dataset_id': dataset['id'],
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

def build_interactive_plotly_and_insights(summary, df=None):
    """Build interactive Plotly chart payloads and automated AI Insights."""
    insights = []
    plotly_data = {}

    target_col = summary.get('target_column', 'Target')
    ptype = summary.get('problem_type', 'classification')
    best_model_info = summary.get('best_model', {})
    best_name = best_model_info.get('name', 'N/A').replace('_', ' ').title()
    best_metrics = best_model_info.get('metrics', {})
    best_score = best_metrics.get('accuracy') if ptype != 'regression' else best_metrics.get('r2')
    if best_score is None:
        best_score = list(best_metrics.values())[0] if best_metrics else 0.0

    # 1. Best Model Insight
    insights.append({
        'icon': 'trophy-fill',
        'color': 'warning',
        'title': 'Winning Algorithm',
        'text': f"<strong>{best_name}</strong> achieved the top score of <strong>{best_score:.4f}</strong> across test samples.",
    })

    # 2. Pipeline Architecture Insight
    feature_cols = summary.get('feature_columns', [])
    train_samples = summary.get('train_samples', 0)
    test_samples = summary.get('test_samples', 0)
    total_samples = train_samples + test_samples
    insights.append({
        'icon': 'cpu-fill',
        'color': 'info',
        'title': 'Pipeline Architecture',
        'text': f"Evaluated <strong>{len(feature_cols)} feature columns</strong> across <strong>{total_samples:,} rows</strong> with automated {ptype} preprocessing.",
    })

    # 3. Model Comparison Plotly Data
    model_scores = summary.get('model_scores', {})
    if model_scores:
        model_names = [k.replace('_', ' ').title() for k in model_scores.keys()]
        scores_vals = [round(float(v), 4) for v in model_scores.values()]
        plotly_data['model_comparison'] = {
            'x': model_names,
            'y': scores_vals,
            'title': 'Model Performance Scores',
        }

    # 4. Feature Health Insight & Correlation
    if df is not None:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty and len(numeric_df.columns) >= 2:
            cols = list(numeric_df.columns[:10])
            corr_matrix = numeric_df[cols].corr().round(2).values.tolist()
            plotly_data['correlation_matrix'] = {
                'z': corr_matrix,
                'x': cols,
                'y': cols,
            }
            insights.append({
                'icon': 'graph-up-arrow',
                'color': 'success',
                'title': 'Feature Correlations',
                'text': f"Generated correlation matrix across <strong>{len(cols)} numerical features</strong> for pattern discovery.",
            })

    return {
        'insights': insights,
        'plotly_data': plotly_data,
    }


@app.route('/dataset/<int:dataset_id>/visualizations')
@login_required
def dataset_visualizations(dataset_id):
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('dashboard'))

    preprocessing_result = db.get_preprocessing_result(dataset_id)
    if not preprocessing_result:
        flash('Run preprocessing before viewing visualizations.', 'warning')
        return redirect(url_for('preprocess', dataset_id=dataset_id))

    summary = json.loads(preprocessing_result['summary_json'] or '{}')
    summary = fix_visualization_paths(summary)

    df = None
    try:
        df = validate_dataset_file(record['file_path'])
    except Exception:
        pass

    ai_pack = build_interactive_plotly_and_insights(summary, df=df)

    return render_template(
        'visualizations.html',
        page_name='dashboard',
        record=dict(record),
        summary=summary,
        visualizations=summary.get('visualizations', {}),
        updated_at=preprocessing_result['preprocessed_at'],
        assistant_history=db.get_assistant_history(g.user['id'], dataset_id),
        ai_insights=ai_pack['insights'],
        plotly_data=ai_pack['plotly_data'],
    )

@app.route('/dataset/<int:dataset_id>/assistant', methods=['POST'])
@login_required
def dataset_assistant(dataset_id):
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        return jsonify({'error': 'Dataset not found.'}), 404

    payload = request.get_json(silent=True) or {}
    question = str(payload.get('question', '')).strip()
    if not question:
        return jsonify({'error': 'Please enter a question.'}), 400
    if len(question) > 2000:
        return jsonify({'error': 'Question is too long.'}), 400

    try:
        df = validate_dataset_file(record['file_path'])
        history = db.get_assistant_history(g.user['id'], dataset_id)
        result = answer_general_question(question)
        if result is None and is_dataset_explanation_request(question):
            preprocessing_result = db.get_preprocessing_result(dataset_id)
            preprocessing_summary = (
                json.loads(preprocessing_result['summary_json'])
                if preprocessing_result and preprocessing_result['summary_json']
                else None
            )
            result = {
                'kind': 'answer',
                'message': explain_dataset(df, record['original_name'], preprocessing_summary),
                'changes': [],
            }
        elif result is None and not is_dataset_update_request(question):
            result = answer_comment_review_question(question, df)
            if result is None:
                result = answer_row_lookup_question(question, df, history=history)
            if result is None:
                result = answer_full_dataset_question(question, df)
            if result is None:
                result = answer_exact_numeric_question(question, df)
        if result is None:
            preprocessing_result = db.get_preprocessing_result(dataset_id)
            preprocessing_summary = (
                json.loads(preprocessing_result['summary_json'])
                if preprocessing_result and preprocessing_result['summary_json']
                else None
            )
            result = request_dataset_assistant(
                question,
                build_assistant_context(df, record['original_name'], question),
                df=df,
                dataset_name=record['original_name'],
                preprocessing_summary=preprocessing_summary,
            )
    except Exception as exc:
        app.logger.exception('Dataset assistant request failed')
        return jsonify({'error': str(exc)}), 503

    if isinstance(result.get('message'), str):
        result['message'] = result['message'].replace('**', '')

    if result.get('kind') == 'update_preview':
        valid_changes = []
        for change in result.get('changes', [])[:20]:
            try:
                row_number = int(change['row_number'])
                column = str(change['column'])
            except (KeyError, TypeError, ValueError):
                continue
            if 1 <= row_number <= len(df) and column in df.columns:
                current = df.iloc[row_number - 1][column]
                old_value = None if pd.isna(current) else str(current)
                proposed_old = change.get('old_value')
                old_matches = (
                    old_value is None and proposed_old in (None, '', 'None', 'nan')
                ) or old_value == str(proposed_old)
                if old_matches:
                    valid_changes.append({
                        'row_number': row_number,
                        'column': column,
                        'old_value': old_value,
                        'new_value': str(change.get('new_value', '')),
                    })
        session[f'assistant_preview_{dataset_id}'] = valid_changes
        result['changes'] = valid_changes
        result['requires_confirmation'] = bool(valid_changes)
    history_response = str(result.get('message', '')).replace('**', '')
    if result.get('requires_confirmation') and result.get('changes'):
        preview = '\n'.join(
            f"Row {change['row_number']}, {change['column']}: "
            f"{change['old_value'] or '(empty)'} -> {change['new_value']}"
            for change in result['changes']
        )
        history_response = f'{history_response}\nUpdate preview:\n{preview}'
    db.save_assistant_history(
        g.user['id'], dataset_id, question, history_response
    )
    return jsonify(result)

@app.route('/dataset/<int:dataset_id>/assistant/confirm', methods=['POST'])
@login_required
def confirm_dataset_assistant_update(dataset_id):
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        return jsonify({'error': 'Dataset not found.'}), 404

    changes = session.pop(f'assistant_preview_{dataset_id}', [])
    if not changes:
        return jsonify({'error': 'No pending update to confirm.'}), 400

    try:
        df = validate_dataset_file(record['file_path'])
        for change in changes:
            row_index = change['row_number'] - 1
            current = df.iloc[row_index][change['column']]
            if (pd.isna(current) and change['old_value'] is not None) or (
                    not pd.isna(current) and str(current) != change['old_value']):
                return jsonify({'error': 'Dataset changed since the preview was created. Please ask again.'}), 409
            df.at[df.index[row_index], change['column']] = change['new_value']

        directory = os.path.dirname(record['file_path'])
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', dir=directory, delete=False, newline='') as handle:
            temp_path = handle.name
            df.to_csv(handle, index=False)
        os.replace(temp_path, record['file_path'])
        db.update_dataset_file_metadata(dataset_id, g.user['id'], len(df), len(df.columns), os.path.getsize(record['file_path']))
    except Exception as exc:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        app.logger.exception('Dataset assistant update failed')
        return jsonify({'error': f'Could not apply update: {exc}'}), 500

@app.route('/dataset/<int:dataset_id>/assistant/clear', methods=['POST'])
@login_required
def clear_assistant_chat_history(dataset_id):
    record = db.get_dataset_by_id(dataset_id, g.user['id'])
    if not record:
        return jsonify({'error': 'Dataset not found.'}), 404
    db.clear_assistant_history(g.user['id'], dataset_id)
    return jsonify({'message': 'Assistant chat history cleared.'})

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Development entry point. For deployment, use a production WSGI server such as Gunicorn.
    app.run(host='127.0.0.1', port=5000)

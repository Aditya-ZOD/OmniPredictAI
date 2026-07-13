import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the heart disease dataset.
df = pd.read_csv('heart.csv')

# Clean missing values ('?') and binarize target variable.
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = pd.to_numeric(df[column], errors='coerce')
df = df.fillna(df.median(numeric_only=True))
for column in ['ca', 'thal']:
    df[column] = pd.to_numeric(df[column], errors='coerce')
df = df.fillna(df.median(numeric_only=True))
df['target'] = df['target'].apply(lambda value: 1 if value > 0 else 0)

X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define candidate model pipelines.
models = {
    'svc': Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=42))
    ]),
    'rf': Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(random_state=42))
    ]),
    'extra': Pipeline([
        ('scaler', StandardScaler()),
        ('extra', ExtraTreesClassifier(random_state=42))
    ]),
    'logreg': Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(max_iter=5000, random_state=42))
    ]),
    'knn': Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ]),
    'gb': Pipeline([
        ('scaler', StandardScaler()),
        ('gb', GradientBoostingClassifier(random_state=42))
    ])
}

params = {
    'svc': {
        'svc__C': [0.5, 1, 2, 5, 10],
        'svc__kernel': ['rbf', 'linear'],
        'svc__gamma': ['scale', 0.01, 0.1, 1]
    },
    'rf': {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [None, 5, 10],
        'rf__min_samples_split': [2, 5]
    },
    'extra': {
        'extra__n_estimators': [200, 400],
        'extra__max_depth': [None, 10, 20],
        'extra__min_samples_split': [2, 5]
    },
    'logreg': {
        'logreg__C': [0.1, 0.5, 1, 2, 5],
        'logreg__solver': ['liblinear', 'lbfgs']
    },
    'knn': {
        'knn__n_neighbors': [3, 5, 7, 9],
        'knn__weights': ['uniform', 'distance']
    },
    'gb': {
        'gb__n_estimators': [50, 100, 200],
        'gb__learning_rate': [0.05, 0.1, 0.2],
        'gb__max_depth': [2, 3, 4]
    }
}

# Search the best hyperparameters for each model.
for name, model in models.items():
    search = GridSearchCV(model, params[name], cv=5, n_jobs=-1, scoring='accuracy')
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    preds = best_model.predict(X_test)
    print(name, 'best', search.best_params_, 'accuracy', round(accuracy_score(y_test, preds) * 100, 2))

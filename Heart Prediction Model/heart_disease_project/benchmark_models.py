import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the dataset

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = [
    ('LogReg', make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))),
    ('DT', make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42))),
    ('RF', make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_estimators=300))),
    ('GBT', make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42))),
    ('ExtraTrees', make_pipeline(StandardScaler(), ExtraTreesClassifier(random_state=42, n_estimators=400))),
    ('AdaBoost', make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=42))),
    ('SVC', make_pipeline(StandardScaler(), SVC(kernel='rbf', C=2.0, gamma='scale', random_state=42))),
    ('KNN', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))),
    ('NB', make_pipeline(StandardScaler(), GaussianNB())),
    ('HistGB', make_pipeline(StandardScaler(), HistGradientBoostingClassifier(random_state=42)))
]

for name, model in models:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(name, round(accuracy_score(y_test, preds) * 100, 2))

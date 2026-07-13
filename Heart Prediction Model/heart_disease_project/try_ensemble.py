import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier

# Load the dataset.
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

estimators = [
    ('logreg', make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))),
    ('svc', make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale', probability=True, random_state=42))),
    ('gnb', make_pipeline(StandardScaler(), GaussianNB())),
    ('lda', make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())),
    ('rf', make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_estimators=300))),
]

for voting in ['hard', 'soft']:
    model = VotingClassifier(estimators=estimators, voting=voting)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(voting, round(accuracy_score(y_test, preds) * 100, 2))

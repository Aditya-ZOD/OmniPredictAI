import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

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

# Single train-test split.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

models = [
    ('LogReg', make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=42))),
    ('DT', make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=42))),
    ('RF', make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42, n_estimators=300))),
    ('ExtraTrees', make_pipeline(StandardScaler(), ExtraTreesClassifier(random_state=42, n_estimators=400))),
    ('GB', make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=42))),
    ('AdaBoost', make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=42))),
    ('SVC', make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42))),
    ('KNN', make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights='distance'))),
    ('GNB', make_pipeline(StandardScaler(), GaussianNB())),
    ('LDA', make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())),
    ('MLP', make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42)))
]

for name, model in models:
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(name, 'test_accuracy', round(accuracy * 100, 2), 'cv_mean', round(cv_scores.mean() * 100, 2), 'cv_std', round(cv_scores.std() * 100, 2))

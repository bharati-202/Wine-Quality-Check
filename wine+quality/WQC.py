import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('winequality-red.csv')
# df = pd.DataFrame(data)
print(data.head())
print(data.info())

print("null values")
print(data.isnull().sum())

print("Describe")
print(data.describe())

data['goodquality'] = [1 if x >=7 else 0 for x in data['quality']]
print(data['goodquality'])

X = data.drop(['quality', 'goodquality'], axis=1)
print(X)
y = data['goodquality']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
lr_pred = log_reg.predict(X_test_scaled)
print("Logistic Regression Accuracy", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))

svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
print("SVM Accuracy", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print(rf_pred)
print('Accuracy', accuracy_score(y_test, rf_pred))
print("Classification Report:", classification_report(y_test, rf_pred))

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train_scaled, y_train)
print("Best SVM parameters", grid.best_params_)
best_svm_pred = grid.predict(X_test_scaled)
print("Turned SVM Accuracy", accuracy_score(y_test, best_svm_pred))
print(classification_report(y_test, best_svm_pred))

cv_scores = cross_val_score(log_reg, X_train_scaled, y_train, cv=5)
print("Logistic Regression CV Mean Accuracy", cv_scores.mean())

feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()
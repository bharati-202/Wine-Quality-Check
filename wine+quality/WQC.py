import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)
print('Accuracy', accuracy_score(y_test, y_pred))
print("Classification Report:", classification_report(y_test, y_pred))


feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()
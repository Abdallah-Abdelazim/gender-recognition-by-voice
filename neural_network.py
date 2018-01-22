import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./input/voice.csv')  # reading the file as a csv

# to check for Nan values
df.isnull().sum()

y = []

# encoding male=1 and female=0
for i in range(len(df.label)):
    if df.label[i] == 'male':
        y.append(1)
    elif df.label[i] == 'female':
        y.append(0)

df = df.drop('label', axis=1)  # drop th ecolumn with labels
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=0)

stdSc = StandardScaler()  # preprocessing
X_train = stdSc.fit_transform(X_train)
X_test = stdSc.fit_transform(X_test)

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30), learning_rate_init=0.001)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))

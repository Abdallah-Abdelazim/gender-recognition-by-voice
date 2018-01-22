import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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

# using logistic regression
C = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
for i in enumerate(C):
    lr = LogisticRegression(C=i[1])
    lr.fit(X_train, y_train)
    print("Logistic Accuracy (C={}): {}".format(i[1], accuracy_score(lr.predict(X_test), y_test)))
print("-----------------------------------------------------------")

# Using SVM
from sklearn import svm

C = [5, 15, 0.01, 0.1, 1, 10, 100, 1000]
kernels = ['rbf', 'linear', 'sigmoid', 'poly']
for i in enumerate(C):
    for k in kernels:
        clf1 = svm.SVC(C=i[1], kernel=k)
        clf1.fit(X_train, y_train)
        print("SVM Accuracy (C={} & kernel={}):".format(i[1], k))
        print(accuracy_score(clf1.predict(X_test), y_test))
    print("-----------------------------------------------------------")

# using k-nearest neighbours
from sklearn.neighbors import KNeighborsClassifier

for i in range(1, 10, 2):  # i skips even numbers
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    print("KNN Accuracy (k={}): {}".format(i, accuracy_score(knn.predict(X_test), y_test)))
print("-----------------------------------------------------------")

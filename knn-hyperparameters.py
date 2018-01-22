import pandas as pd
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

# knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV

model = KNeighborsClassifier()

k_range = list(range(1, 101))
weight_options = ['uniform', 'distance']

param_dist = dict(n_neighbors=k_range, weights=weight_options)

rand = RandomizedSearchCV(estimator=model, cv=10, param_distributions=param_dist, n_iter=10)
rand.fit(X_train, y_train)

print(rand.best_score_)
print(rand.best_params_)
print("KNN Accuracy (k={}): {}".format(rand.best_params_['n_neighbors'], rand.best_score_))

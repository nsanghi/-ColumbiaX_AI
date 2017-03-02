import csv
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


if __name__ == '__main__':

  X = []
  y = []
  with open(sys.argv[1], 'rU') as f1:
    inputreader = csv.reader(f1)
    for row in inputreader:
      X.append([float(row[0]), float(row[1])])
      y.append(float(row[2]))

    X = np.array(X)
    y = np.array(y)

    #X_scaled = preprocessing.scale(X)
    X_scaled = X

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, stratify=y, random_state=123)
    # print sum(y_train==0), sum(y_train==1), sum(y_test==0), sum(y_test==1)

    # probem 1 to 3
    #tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
    tuned_parameters = [{'kernel': ['poly'], 'C' : [0.1, 1., 3.], 'degree' : [4, 5, 6], 'gamma' : [0.1, 1.]}]
    tuned_parameters = [{'C' : [0.1, 1., 3.], 'degree' : [4, 5, 6], 'gamma' : [0.1, 1.]}]
    #tuned_parameters = [{'kernel': ['rbf'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]}]
    clf = GridSearchCV(SVC(kernel='poly'), tuned_parameters, cv=5, scoring='accuracy')

    # problem 4
    #tuned_parameters = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
    #clf = GridSearchCV(LogisticRegression(), tuned_parameters, cv=5, scoring='accuracy')

    # problem 5
    #n_neighbors = np.linspace(1,50,50).astype(np.int8)
    #leaf_size = np.linspace(5,60,12).astype(np.int8)
    #tuned_parameters = [{'n_neighbors': n_neighbors, 'leaf_size':leaf_size}]
    #clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='accuracy')

    # problem 6
    #max_depth = np.linspace(1,50,50).astype(np.int8)
    #min_samples_split = [2,3,4,5,6,7,8,9,10]
    #tuned_parameters = [{'max_depth': max_depth, 'min_samples_split':min_samples_split}]
    #clf = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy')

    # problem 7
    #max_depth = np.linspace(1,50,50).astype(np.int8)
    #min_samples_split = [2,3,4,5,6,7,8,9,10]
    #tuned_parameters = [{'max_depth': max_depth, 'min_samples_split':min_samples_split}]
    #clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy')

    clf.fit(X_train, y_train)
    print clf.best_score_
    print accuracy_score(y_test, clf.predict(X_test), normalize=True)
    print clf.best_params_


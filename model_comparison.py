# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:16:30 2021

@author: HP
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


X = pd.read_csv('deGrom_data_clean.csv')
y = pd.read_csv('deGrom_data_class.csv')

idx = np.random.RandomState(seed=42).permutation(X.index)
X = X.reindex(idx).reset_index(drop=True)
y = y.reindex(idx).reset_index(drop=True)

X = X.to_numpy()
y = np.ravel(y.to_numpy())

classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis()]

for clf in classifiers:
    print(clf)
    cvs = cross_val_score(clf,X,y)
    print(cvs)
    print(np.mean(cvs))
    print('')

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 22:16:30 2021

@author: HP
"""

# from IPython import get_ipython
# get_ipython().magic('reset -sf')


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn import feature_selection
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

random_state=0

X = pd.read_csv('data/deGrom_data_clean.csv')
y = pd.read_csv('data/deGrom_data_class.csv')

idx = np.random.RandomState(seed=42).permutation(X.index)
X = X.reindex(idx).reset_index(drop=True)
y = y.reindex(idx).reset_index(drop=True)

X = X.to_numpy()
y = np.ravel(y.to_numpy())

# preprocess
prep = [
    'passthrough',
    preprocessing.StandardScaler(),
    preprocessing.MinMaxScaler(),
    preprocessing.QuantileTransformer(random_state=random_state), # to uniform distribution
    # preprocessing.PowerTransformer(method='box-cox', standardize=False), # to gaussian distribution
    preprocessing.Normalizer(norm='l1'),
    preprocessing.Normalizer(norm='l2'),
    preprocessing.PolynomialFeatures(2),
    decomposition.PCA(whiten=True),
    decomposition.FastICA(whiten=True),
]

# feature engineering
feature_1 = [
    'passthrough',
    # feature_selection.VarianceThreshold(threshold=(.95 * (1 - .95))),
    # feature_selection.SelectFromModel(LinearSVC(penalty='l1')),
    # feature_selection.SelectFromModel(ExtraTreesClassifier(n_estimators=20))
]

feature_2 = [
    'passthrough',
    # decomposition.PCA(n_components=int(0.5 * X.shape[1])),
    decomposition.PCA(n_components=int(0.5 * X.shape[1])),

    # decomposition.KernelPCA(n_components=int(0.8 * X.shape[1]), kernel='rbf'),
    # decomposition.KernelPCA(kernel='rbf'),
    # decomposition.KernelPCA(n_components=int(0.8 * X.shape[1]), kernel='poly'),
    # decomposition.KernelPCA(kernel='poly'),
    # decomposition.FastICA(n_components=int(0.5 * X.shape[1])),
    decomposition.FastICA(n_components=int(0.5 * X.shape[1])),
    # decomposition.LatentDirichletAllocation(n_components=int(0.3 * X.shape[1]))
]

# classifiers
classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(max_depth=5),
    AdaBoostClassifier(n_estimators=20),
    RandomForestClassifier(max_depth=None, n_estimators=10, max_features=1),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    LinearDiscriminantAnalysis(),
    # SVC(kernel='linear'),
    # SVC(kernel='rbf')
    ]

result = open("eval_result.tsv", "w", encoding="utf-8")


for p in prep:
    for feat1 in feature_1:
        for feat2 in feature_2:
            for clf in classifiers:
                print("*" * 50)
                print("preprocess: {}, feature_selection: {}, dim_reduction: {}, classifier: {}".format(p, feat1, feat2, clf), flush=True)
                
                estimators = [
                    ('preprocess', p),
                    ('feature_selection', feat1),
                    ('dim_reduction', feat2),
                    ('classifier', clf)
                ]
                pipeline = Pipeline(estimators)
                
                cvs = cross_val_score(pipeline, X, y)
                print("score: {}".format(np.mean(cvs)))
                print("{}\t{}\t{}\t{}\t{}".format(type(p).__name__, type(feat1).__name__, type(feat2).__name__, type(clf).__name__, np.mean(cvs)), file=result, flush=True)
                print("*" * 50, flush=True)

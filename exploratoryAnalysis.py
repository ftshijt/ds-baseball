# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 16:34:11 2021

@author: HP
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

X = pd.read_csv('deGrom_data_clean.csv')
C = pd.read_csv('deGrom_data_class.csv')

idx = np.random.RandomState(seed=42).permutation(X.index)
X = X.reindex(idx).reset_index(drop=True)
C = C.reindex(idx).reset_index(drop=True)

X = X.to_numpy()
C = np.ravel(C.to_numpy())

pca = PCA(n_components=X.shape[1]).fit(X)

plt.figure(1)
plt.subplot(211)
plt.plot(pca.explained_variance_,'ro-')

plt.subplot(212)
cmsm = np.cumsum(pca.explained_variance_)
plt.plot(cmsm/cmsm[-1],'bs-')

plt.figure(2)
plt.matshow(pca.get_covariance())


from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.ensemble import RandomForestClassifier as RFC


clf = LDA()
cvs = cross_val_score(clf,X,C)
print(cvs)


























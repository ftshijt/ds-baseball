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

X = pd.read_csv('deGrom_data_clean.csv')
y = pd.read_csv('deGrom_data_class.csv')
#print(np.shape(X))

idx = np.random.RandomState(seed=420).permutation(X.index)
X = X.reindex(idx).reset_index(drop=True)
y = y.reindex(idx).reset_index(drop=True)

X = X.to_numpy()
y = np.ravel(y.to_numpy())
print(np.shape(y))

##################################################################

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn_extra.cluster import KMedoids

it, yt = [], []
for k in range(1,Xtrain.shape[1]):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(Xtrain)
    i = kmeans.inertia_
    yt.append(i)
    it.append(k)
plt.plot(it,yt)
plt.show()
###################################################################
cluster_obj = [
    KMeans(n_clusters=6, random_state=0),
    GaussianMixture(n_components=2, random_state=0),
    SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0)]

###################################################################
recent_cluster_num = 15000
XC = X[-recent_cluster_num:]
yC = y[-recent_cluster_num:]

num_test = 60
a_class, a_kmeans, a_kmedians, a_gmm, a_spectral, test_count = [], [], [], [], [], []
for m in range(1, num_test+1):
    classification_pred, clustering_pred_kmeans, clustering_pred_kmedians, clustering_pred_gmm, clustering_pred_spectral, real = [], [], [], [], [], []
    XC = X[-recent_cluster_num:]
    yC = y[-recent_cluster_num:]
    for i in range(1, m+1):
        kmeans = KMeans(n_clusters=5, random_state=0).fit(XC)
        kmedians = KMedoids(n_clusters=5, random_state=0).fit(XC)
        gmm = GaussianMixture(n_components=5, random_state=0).fit(XC)
        XC_p = X[recent_cluster_num+i,:].reshape(1,27)
        label_kmeans = kmeans.predict(XC_p)
        label_kmedians = kmedians.predict(XC_p)
        label_gmm = gmm.predict(XC_p)
        centroid_kmeans = kmeans.cluster_centers_[label_kmeans]
        centroid_kmedians = kmedians.cluster_centers_[label_kmedians]
        centroid_gmm = gmm.means_[label_gmm]
        clf = LinearDiscriminantAnalysis()
        class_p = clf.fit(X,y).predict(XC_p)
        clus_p_kmeans = clf.fit(XC,yC).predict(centroid_kmeans)
        clus_p_kmedians = clf.fit(XC,yC).predict(centroid_kmedians)
        clus_p_gmm = clf.fit(XC,yC).predict(centroid_gmm)
        classification_pred.append(class_p)
        clustering_pred_kmeans.append(clus_p_kmeans)
        clustering_pred_kmedians.append(clus_p_kmedians)
        clustering_pred_gmm.append(clus_p_gmm)
        real.append(y[recent_cluster_num+i])
    
        XCC = np.append(XC,X[recent_cluster_num+i,:]).reshape(recent_cluster_num+i,27)
        spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', random_state=0).fit_predict(XCC)
        spec_points = XCC[spectral == spectral[-1]]
        centroid_spectral = spec_points.mean(axis=0)
        centroid_spectral = centroid_spectral.reshape(1,27)
        clus_p_spectral = clf.fit(XC,yC).predict(centroid_spectral)
        clustering_pred_spectral.append(clus_p_spectral)    
    
        XC = np.append(XC,X[recent_cluster_num+i,:]).reshape(recent_cluster_num+i,27)
        yC = np.append(yC,y[recent_cluster_num+i])

    classification_pred = np.array(classification_pred)
    clustering_pred_kmeans = np.array(clustering_pred_kmeans)
    clustering_pred_kmedians = np.array(clustering_pred_kmedians)
    clustering_pred_gmm = np.array(clustering_pred_gmm)
    clustering_pred_spectral = np.array(clustering_pred_spectral)
    real = (np.array(real)).reshape(m,1)

    accuracy_class = np.sum(classification_pred == real)/m*100
    accuracy_clus_kmeans = (np.sum(clustering_pred_kmeans == real))/m*100
    accuracy_clus_kmedians = (np.sum(clustering_pred_kmedians == real))/m*100
    accuracy_clus_gmm = (np.sum(clustering_pred_gmm == real))/m*100
    accuracy_clus_spectral = (np.sum(clustering_pred_spectral == real))/m*100

    print(accuracy_class, accuracy_clus_kmeans, accuracy_clus_kmedians, accuracy_clus_gmm, accuracy_clus_spectral)
    a_class.append(accuracy_class)
    a_kmeans.append(accuracy_clus_kmeans)
    a_kmedians.append(accuracy_clus_kmedians)
    a_gmm.append(accuracy_clus_gmm)
    a_spectral.append(accuracy_clus_spectral)
    test_count.append(m)
    
plt.plot(test_count,a_class,'k')
plt.plot(test_count,a_kmeans,'b')
plt.plot(test_count,a_kmedians,'r')
plt.plot(test_count,a_gmm,'g')
plt.plot(test_count,a_spectral,'y')
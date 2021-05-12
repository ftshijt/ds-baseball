# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:12:56 2021

@author: HP
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


X = pd.read_csv('deGrom_data_clean.csv')
y = pd.read_csv('deGrom_data_class.csv')

df = X.copy()
df['pitch_class'] = y.copy()


###############################################################################
# feature importance

features = X.columns.copy()

idx = np.random.RandomState(seed=42).permutation(X.index)
X = X.reindex(idx).reset_index(drop=True)
y = y.reindex(idx).reset_index(drop=True)

X = X.to_numpy()
y = np.ravel(y.to_numpy())

clf = RandomForestClassifier(max_depth=None, n_estimators=100, max_features=1, random_state=42)

clf.fit(X,y)

importances = clf.feature_importances_
std = np.std([
    tree.feature_importances_ for tree in clf.estimators_], axis=0)

df1 = pd.DataFrame(importances,index=features,columns=['Importance'])
df1 = df1.rename_axis('Feature')
df1['STD'] = std
df1.sort_values(by=['Importance'],inplace=True,ascending=True)

plt.figure(1,figsize=(15,10),dpi=200)
plt.barh(df1.index,df1.Importance,xerr=df1.STD,
         color=plt.cm.cool(75),
         edgecolor=None,
         ecolor=plt.cm.cool(255),
         capsize=3)



###############################################################################
# analysis by pitch number
# adapted from 
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/horizontal_barchart_distribution.html#sphx-glr-gallery-lines-bars-and-markers-horizontal-barchart-distribution-py

df2 = df[['pitch_number','pitch_class']].copy()
df2.drop(df2.loc[df2['pitch_class']==6].index, axis=0, inplace=True)

# most used pitches in order
# 2,1,3,5,4
pitchDict = {'SL':1,'FF':2,'CH':3,'CU':4,'FT':5}
total = np.array([np.sum(df2['pitch_class']==2),
                  np.sum(df2['pitch_class']==1),
                  np.sum(df2['pitch_class']==3),
                  np.sum(df2['pitch_class']==5),
                  np.sum(df2['pitch_class']==4)])

pitch01 = df2[df2['pitch_number']==1].copy()
pitch02 = df2[df2['pitch_number']==2].copy()
pitch03 = df2[df2['pitch_number']==3].copy()
pitch04 = df2[df2['pitch_number']==4].copy()
pitch05 = df2[df2['pitch_number']==5].copy()
pitch06 = df2[df2['pitch_number']==6].copy()
pitch07 = df2[df2['pitch_number']==7].copy()
pitch08 = df2[df2['pitch_number']==8].copy()
pitch09 = df2[df2['pitch_number']==9].copy()
pitch10 = df2[df2['pitch_number']==10].copy()
pitch11 = df2[df2['pitch_number']==11].copy()
pitch12 = df2[df2['pitch_number']==12].copy()
pitch13 = df2[df2['pitch_number']==13].copy()

p01 = np.array([np.sum(pitch01['pitch_class']==2),
                np.sum(pitch01['pitch_class']==1),
                np.sum(pitch01['pitch_class']==3),
                np.sum(pitch01['pitch_class']==5),
                np.sum(pitch01['pitch_class']==4)])

p02 = np.array([np.sum(pitch02['pitch_class']==2),
                np.sum(pitch02['pitch_class']==1),
                np.sum(pitch02['pitch_class']==3),
                np.sum(pitch02['pitch_class']==5),
                np.sum(pitch02['pitch_class']==4)])

p03 = np.array([np.sum(pitch03['pitch_class']==2),
                np.sum(pitch03['pitch_class']==1),
                np.sum(pitch03['pitch_class']==3),
                np.sum(pitch03['pitch_class']==5),
                np.sum(pitch03['pitch_class']==4)])

p04 = np.array([np.sum(pitch04['pitch_class']==2),
                np.sum(pitch04['pitch_class']==1),
                np.sum(pitch04['pitch_class']==3),
                np.sum(pitch04['pitch_class']==5),
                np.sum(pitch04['pitch_class']==4)])

p05 = np.array([np.sum(pitch05['pitch_class']==2),
                np.sum(pitch05['pitch_class']==1),
                np.sum(pitch05['pitch_class']==3),
                np.sum(pitch05['pitch_class']==5),
                np.sum(pitch05['pitch_class']==4)])

p06 = np.array([np.sum(pitch06['pitch_class']==2),
                np.sum(pitch06['pitch_class']==1),
                np.sum(pitch06['pitch_class']==3),
                np.sum(pitch06['pitch_class']==5),
                np.sum(pitch06['pitch_class']==4)])

p07 = np.array([np.sum(pitch07['pitch_class']==2),
                np.sum(pitch07['pitch_class']==1),
                np.sum(pitch07['pitch_class']==3),
                np.sum(pitch07['pitch_class']==5),
                np.sum(pitch07['pitch_class']==4)])

p08 = np.array([np.sum(pitch08['pitch_class']==2),
                np.sum(pitch08['pitch_class']==1),
                np.sum(pitch08['pitch_class']==3),
                np.sum(pitch08['pitch_class']==5),
                np.sum(pitch08['pitch_class']==4)])

p09 = np.array([np.sum(pitch09['pitch_class']==2),
                np.sum(pitch09['pitch_class']==1),
                np.sum(pitch09['pitch_class']==3),
                np.sum(pitch09['pitch_class']==5),
                np.sum(pitch09['pitch_class']==4)])

p10 = np.array([np.sum(pitch10['pitch_class']==2),
                np.sum(pitch10['pitch_class']==1),
                np.sum(pitch10['pitch_class']==3),
                np.sum(pitch10['pitch_class']==5),
                np.sum(pitch10['pitch_class']==4)])

p11 = np.array([np.sum(pitch11['pitch_class']==2),
                np.sum(pitch11['pitch_class']==1),
                np.sum(pitch11['pitch_class']==3),
                np.sum(pitch11['pitch_class']==5),
                np.sum(pitch11['pitch_class']==4)])

p12 = np.array([np.sum(pitch12['pitch_class']==2),
                np.sum(pitch12['pitch_class']==1),
                np.sum(pitch12['pitch_class']==3),
                np.sum(pitch12['pitch_class']==5),
                np.sum(pitch12['pitch_class']==4)])

p13 = np.array([np.sum(pitch13['pitch_class']==2),
                np.sum(pitch13['pitch_class']==1),
                np.sum(pitch13['pitch_class']==3),
                np.sum(pitch13['pitch_class']==5),
                np.sum(pitch13['pitch_class']==4)])

total_n = total/np.sum(total)
p01_n = p01/np.sum(p01)
p02_n = p02/np.sum(p02)
p03_n = p03/np.sum(p03)
p04_n = p04/np.sum(p04)
p05_n = p05/np.sum(p05)
p06_n = p06/np.sum(p06)
p07_n = p07/np.sum(p07)
p08_n = p08/np.sum(p08)
p09_n = p09/np.sum(p09)
p10_n = p10/np.sum(p10)
p11_n = p11/np.sum(p11)
p12_n = p12/np.sum(p12)
p13_n = p13/np.sum(p13)


results = {'Total':total_n,
           'Pitch_1':p01_n,
           'Pitch_2':p02_n,
           'Pitch_3':p03_n,
           'Pitch_4':p04_n,
           'Pitch_5':p05_n,
           'Pitch_6':p06_n,
           'Pitch_7':p07_n,
           'Pitch_8':p08_n,
           'Pitch_9':p09_n,
           'Pitch_10':p10_n,
           'Pitch_11':p11_n,
           'Pitch_12':p12_n,
           'Pitch_13':p13_n}

category_names = ['4-seam Fastball', 
                  'Slider',
                  'Changeup', 
                  '2-seam Fastball', 
                  'Curveball']


def pitches(results, category_names):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*.
    category_names : list of str
        The category labels.
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('cool')(
        np.linspace(0.15, 0.85, data.shape[1]))

    #RdYlGn

    fig, ax = plt.subplots(figsize=(15,10),dpi=200)
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        
        ax.bar_label(rects, fmt='%.2f', label_type='center', color=text_color)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0.125, 1),
              loc='lower left', fontsize='large')

    return fig, ax


pitches(results, category_names)
plt.show()













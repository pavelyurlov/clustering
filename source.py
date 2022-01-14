from algorithm import *

import numpy as np
import pandas as pd
import os
import tqdm
import time
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns




def MARE(true, pred):
    return abs(true - pred) / true


def find_best_k(X, krange=range(1, 50)):
    inertia = []
    for k in krange:
        kmeans = KMeans(n_clusters=k, init='random').fit(X)
        k_inertia = kmeans.inertia_
        inertia.append(k_inertia)
    inertia = np.array(inertia)

    # elbow2 criterion
    best_k_e2, max_index_e2 = 0, -1
    for i in range(2, len(inertia) - 2):
        k = krange[i]
        index = (inertia[i - 2] - inertia[i]) / (inertia[i] - inertia[i + 2])
        if index > max_index_e2:
            best_k_e2 = k
            max_index_e2 = index
    labels_e2 = KMeans(n_clusters=best_k_e2, init='random').fit_predict(X)

    # c-h criterion
    best_k_ch, max_index_ch = 0, -1
    N = len(X)
    for i in range(1, len(inertia)):
        k = krange[i]
        index = (inertia[0] - inertia[i]) / inertia[i] * (N - k) / (k - 1)
        if index > max_index_ch:
            best_k_ch = k
            max_index_ch = index
    labels_ch = KMeans(n_clusters=best_k_ch, init='random').fit_predict(X)
    return (labels_e2, best_k_e2), (labels_ch, best_k_ch)


def apply(data=None, configuration=None, algorithm=None, krange=range(1, 50), verbose=False, show=True):
    results = None
    if algorithm == 'paper':
        results = pd.DataFrame(columns=['clusters',
                                        'paper pred. clusters', 'paper ARI', 'paper MARE'])
    elif algorithm == 'kmeans':
        results = pd.DataFrame(columns=['clusters',
                                        'kmeans elbow2 pred. clusters', 'kmeans elbow2 ARI', 'kmeans elbow2 MARE',
                                        'kmeans c-h pred. clusters', 'kmeans c-h ARI', 'kmeans c-h MARE'])
    else:
        raise ValueError
    datasets = []
    parent_path = ''
    if data == 'real':
        parent_path = 'datasets/real'
    elif data == 'synthetic' and configuration is not None:
        parent_path = 'datasets/synthetic/{}'.format(configuration)
    else:
        raise ValueError
    datasets = sorted(os.listdir(parent_path))
    for ds in tqdm.tqdm(datasets):
        folder_path = '{}/{}/'.format(parent_path, ds)
        X = pd.read_csv(folder_path + 'X.csv', header=None).values.astype(float)
        y = pd.read_csv(folder_path + 'y.csv', header=None).values[:, 0].astype(int)
        y_size = len(set(list(y)))
        row_data, row_name = None, None
        if algorithm == 'paper':
            C, p = Clustering().fit_predict(X, verbose=verbose)
            ari = adjusted_rand_score(y, C)
            mare = MARE(y_size, p)
            row_data = [[y_size, p, ari, mare]]
        elif algorithm == 'kmeans':
            (C_e2, p_e2), (C_ch, p_ch) = find_best_k(X, krange)
            ari_e2 = adjusted_rand_score(y, C_e2)
            mare_e2 = MARE(y_size, p_e2)
            ari_ch = adjusted_rand_score(y, C_ch)
            mare_ch = MARE(y_size, p_ch)
            row_data = [[y_size, p_e2, ari_e2, mare_e2, p_ch, ari_ch, mare_ch]]
        if data == 'real':
            row_name = ds
        else:
            row_name = '{}:{}'.format(configuration, ds)
        row_df = pd.DataFrame(row_data, index=pd.Index([row_name]), columns=results.columns)
        results = results.append(row_df)
    summary_data = [results.mean(), results.std()]
    summary_index = pd.Index(['mean', 'std'])
    summary_columns = results.columns
    summary_df = pd.DataFrame(summary_data, summary_index, summary_columns)
    results = results.append(summary_df)
    if show:
        display(results)
    return results


'''
def show(results):
    for name, row in results.iterrows():
        if row['f'] is None:
            break
        X, y, z, p = row['X'], row['y'], row['pred. y'], row['pred. № of clusters']
        print(name, len(set(y)), len(set(z)), p)
        f, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=axes[0])
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=z, ax=axes[1])
        f.show()
    return
'''

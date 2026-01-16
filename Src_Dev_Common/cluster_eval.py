#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np

from itertools import combinations

def complete_diameter_distance(X):
    res = []
    for i, j in combinations(range(X.shape[0]),2):
        a_i = X[i, :]
        a_j = X[j, :]
        res.append(np.linalg.norm(a_i-a_j))

    return np.max(res)

def average_diameter_distance(X):
    res = []
    for i, j in combinations(range(X.shape[0]),2):
        a_i = X[i, :]
        a_j = X[j, :]
        res.append(np.linalg.norm(a_i-a_j))

    return np.mean(res)

def centroid_diameter_distance(X):
    center = np.mean(X, axis=0)
    res = 2*np.mean([np.linalg.norm(x-center) for x in X])

    return res

def single_linkage_distance(X1, X2):
    res = []
    for x1 in X1:
        for x2 in X2:
            res.append(np.linalg.norm(x1-x2))
    return np.min(res)

def complete_linkage_distance(X1, X2):
    res = []
    for x1 in X1:
        for x2 in X2:
            res.append(np.linalg.norm(x1-x2))
    return np.max(res)

def average_linkage_distance(X1, X2):
    res = []
    for x1 in X1:
        for x2 in X2:
            res.append(np.linalg.norm(x1-x2))
    return np.mean(res)

def centroid_linkage_distance(X1, X2):
    center1 = np.mean(X1, axis=0)
    center2 = np.mean(X2, axis=0)
    return np.linalg.norm(center1-center2)

def average_of_centroids_linkage_distance(X1, X2):
    center1 = np.mean(X1, axis=0)
    center2 = np.mean(X2, axis=0)
    res = []
    for x1 in X1:
        res.append(np.linalg.norm(x1-center2))
    for x2 in X2:
        res.append(np.linalg.norm(x2-center1))
        
    return np.mean(res)

def get_Dunn_index(X, labels, intra_cluster_distance_type='cmpl_dd', 
                   inter_cluster_distance_type='av_cent_ld'):

    intra_cdt_dict = {
        'cmpl_dd':complete_diameter_distance,
        'avdd' : average_diameter_distance,
        'cent_dd' : centroid_diameter_distance
    }
    inter_cdt_dict = {
        'sld' : single_linkage_distance,
        'cmpl_ld' :complete_linkage_distance,
        'avld': average_linkage_distance,
        'cent_ld' : centroid_linkage_distance, 
        'av_cent_ld' : average_of_centroids_linkage_distance
    }
    # intra cluster distance
    intra_cluster_distance = intra_cdt_dict[intra_cluster_distance_type] 

    # inter cluster distance
    inter_cluster_distance = inter_cdt_dict[inter_cluster_distance_type]

    # get minimum value of inter_cluster_distance
    res1 = []
    for i, j in combinations(np.unique(labels),2):
        X1 = X[np.where(labels==i)[0], :]
        X2 = X[np.where(labels==j)[0], :]
        res1.append(inter_cluster_distance(X1, X2))
    min_inter_cd = np.min(res1)

    # get maximum value of intra_cluser_distance

    res2 = []
    for label in np.unique(labels):
        X_target = X[np.where(labels==label)[0], :]
        if X_target.shape[0] >= 2:
            res2.append(intra_cluster_distance(X_target))
        else:
            res2.append(0)
    max_intra_cd = np.max(res2)

    Dunn_idx = min_inter_cd/max_intra_cd
    return Dunn_idx

def get_silhouette_results(X, labels):
    def get_sum_distance(target_x, target_cluster):
        res = np.sum([np.linalg.norm(target_x-x) for x in target_cluster])
        return res
    
    '''
    각 데이터 포인트를 돌면서 a(i), b(i)를 계산
    그리고 s(i)를 계산한다.
    
    마지막으로 Silhouette(실루엣) Coefficient를 계산한다.
    '''
    uniq_labels = np.unique(labels)
    silhouette_val_list = []
    for i in range(len(labels)):
        target_data = X[i]

        ## calculate a(i)
        target_label = labels[i]
        target_cluster_data_idx = np.where(labels==target_label)[0]
        if len(target_cluster_data_idx) == 1:
            silhouette_val_list.append(0)
            continue
        else:
            target_cluster_data = X[target_cluster_data_idx]
            temp1 = get_sum_distance(target_data, target_cluster_data)
            a_i = temp1/(target_cluster_data.shape[0]-1)

        ## calculate b(i)
        b_i_list = []
        label_list = uniq_labels[np.unique(labels) != target_label]
        for ll in label_list:
            other_cluster_data_idx = np.where(labels==ll)[0]
            other_cluster_data = X[other_cluster_data_idx]
            temp2 = get_sum_distance(target_data, other_cluster_data)
            temp_b_i = temp2/other_cluster_data.shape[0]
            b_i_list.append(temp_b_i)

        b_i = min(b_i_list)
        s_i = (b_i-a_i)/max(a_i, b_i)
        silhouette_val_list.append(s_i)

    silhouette_coef_list = []
    for ul in uniq_labels:
        temp3 = np.mean([s for s, l in zip(silhouette_val_list, labels) if l == ul])
        silhouette_coef_list.append(temp3)

    silhouette_coef = max(silhouette_coef_list)
    return (silhouette_coef, np.array(silhouette_val_list))


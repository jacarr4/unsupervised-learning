from data_loader import Dataset, get_dataset, best_params

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, ShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA, FastICA
from time import time
from sklearn.random_projection import GaussianRandomProjection
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import scale

# def do_evaluation( reduction, dataset ):
#     _, _, _, X, y = get_dataset( dataset )
#     test_size = 0.2
#     n_components = best_params[ dataset ]

#     if reduction is None:
#         X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = test_size )
#     elif reduction == 'pca':
#         reduced_X = PCA( n_components = n_components ).fit_transform( X )
#         X_train, X_test, y_train, y_test = train_test_split( reduced_X, y, test_size = test_size )
#     elif reduction == 'ica':
#         reduced_X = FastICA( n_components = n_components ).fit_transform( X )
#         X_train, X_test, y_train, y_test = train_test_split( reduced_X, y, test_size = test_size )
#     elif reduction == 'rp':
#         reduced_X = GaussianRandomProjection( n_components = n_components ).fit_transform( X )
#         X_train, X_test, y_train, y_test = train_test_split( reduced_X, y, test_size = test_size )
#     elif reduction == 'vt':
#         reduced_X = VarianceThreshold( threshold = 1 ).fit_transform( X )
#         X_train, X_test, y_train, y_test = train_test_split( reduced_X, y, test_size = test_size )
#     else:
#         raise ValueError( 'invalid reduction alg' )

#     clf = MLPClassifier()
#     t0 = time()
#     clf.fit(X_train, y_train)
#     score = clf.score( X_test, y_test )
#     total_time = time() - t0

#     return score, total_time

def cluster_features(self):
    data = pd.read_csv('diabetes.csv')
    n_clusters = 4
    km = KMeans(n_clusters=n_clusters).fit(data)

    cluster_map = pd.DataFrame()
    cluster_map['data_index'] = data.index.values
    cluster_map['cluster'] = km.labels_
    print(data.values[0:10])
    new_map = {}
    print(cluster_map)
    X = []
    y = []
    for data_idx, cluster_label in zip(cluster_map['data_index'], cluster_map['cluster']):
        # print("data index = " + str(data_idx))
        # print("cluster label = " + str(cluster_label))
        # if cluster_label not in new_map:
        X.append([1 if i == cluster_label else 0 for i in range(n_clusters) ])
        y.append(data.values[int(data_idx)][8])
        # else:
        #     new_map[cluster_label].append(data.values[int(data_idx)])

    # print(new_map)
    print(X)
    return X, y

def do_evaluation(X, y):
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2 )

    clf = MLPClassifier()
    t0 = time()
    clf.fit(X_train, y_train)
    score = clf.score( X_test, y_test )
    total_time = time() - t0

    return score, total_time

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset', action = 'store', dest = 'dataset', required = True )
    args = parser.parse_args()

    # Learning graphs
    # dataset = Dataset[ args.dataset ]
    # compute_learning_graphs( dataset )

    # Table
    dataset = Dataset[ args.dataset ]

    _, _, _, data, labels = get_dataset( dataset )

    kmeans = KMeans( init='k-means++', n_clusters = 4, n_init=10 )
    kmeans.fit( data )

    # print( kmeans.labels_ )
    # print( data )
    # df = pd.DataFrame( data ) 
    # df.columns = [ [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ] ]
    # df[ 'Cluster' ] = kmeans.labels_
    # print( df )

    # df = pd.DataFrame( [ 'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3' ] )
    df = pd.DataFrame( [], columns = [ 'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3' ] )
    print( df )
    # df.columns = [ [ 'Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3' ] ]
    for l in kmeans.labels_:
        if l == 0:
            df = df.append( { 'Cluster 0': 1, 'Cluster 1': 0, 'Cluster 2': 0, 'Cluster 3': 0 }, ignore_index = True )
        elif l == 1:
            df = df.append( { 'Cluster 0': 0, 'Cluster 1': 1, 'Cluster 2': 0, 'Cluster 3': 0 }, ignore_index = True )
        elif l == 2:
            df = df.append( { 'Cluster 0': 0, 'Cluster 1': 0, 'Cluster 2': 1, 'Cluster 3': 0 }, ignore_index = True )
        elif l == 3:
            df = df.append( { 'Cluster 0': 0, 'Cluster 1': 0, 'Cluster 2': 0, 'Cluster 3': 1 }, ignore_index = True )
    
    X = scale( df )
    y = labels

    score, total_time = do_evaluation( X, y )
    print( 'Score with KMeans clusters as features: %s (took %.2fs)' % ( score, total_time ) )

    # X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2 )
    # clf = MLPClassifier()
    # clf.fit( X_train, y_train )
    # print( clf.score( X_test, y_test ) )
    # print( df )

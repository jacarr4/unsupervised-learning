from data_loader import Dataset, get_dataset

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from enum import IntEnum, auto
import argparse
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture

np.random.seed(42)

sample_size = 300

class Metric( IntEnum ):
    Homogeneity  = auto()
    Completeness = auto()
    V_measure    = auto()
    Rand         = auto()
    Mutual_info  = auto()
    Silhouette   = auto()

count = 0

def get_kmeans_metric( metric: Metric, estimator, data, labels ):
    global count
    print( '%d%%' % (100*count/107) )
    count += 1
    if metric == Metric.Homogeneity:
        return metrics.homogeneity_score(labels, estimator.labels_)
    elif metric == Metric.Completeness:
        return metrics.completeness_score(labels, estimator.labels_)
    elif metric == Metric.V_measure:
        return metrics.v_measure_score(labels, estimator.labels_)
    elif metric == Metric.Rand:
        return metrics.adjusted_rand_score(labels, estimator.labels_)
    elif metric == Metric.Mutual_info:
        return metrics.adjusted_mutual_info_score(labels,  estimator.labels_)
    elif metric == Metric.Silhouette:
        return metrics.silhouette_score(data, estimator.labels_,
                                        metric='euclidean',
                                        sample_size=sample_size)

def get_em_metric( metric: Metric, estimator, data, labels ):
    cluster_labels = estimator.predict( data )
    global count
    print( '%d%%' % (100*count/107) )
    count += 1
    if metric == Metric.Homogeneity:
        return metrics.homogeneity_score(labels, cluster_labels)
    elif metric == Metric.Completeness:
        return metrics.completeness_score(labels, cluster_labels)
    elif metric == Metric.V_measure:
        return metrics.v_measure_score(labels, cluster_labels)
    elif metric == Metric.Rand:
        return metrics.adjusted_rand_score(labels, cluster_labels)
    elif metric == Metric.Mutual_info:
        return metrics.adjusted_mutual_info_score(labels, cluster_labels)
    elif metric == Metric.Silhouette:
        return metrics.silhouette_score(data, cluster_labels,
                                        metric = 'euclidean',
                                        sample_size = sample_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset', action = 'store', dest = 'dataset', required = True )
    parser.add_argument( '--clustering', action = 'store', dest = 'clustering', required = True )
    args = parser.parse_args()

    dataset = Dataset[ args.dataset ]
    n_samples, n_features, n_labels, data, labels = get_dataset( dataset )

    N = 20

    plt.xticks(range(2,N))

    # kmeans = KMeans( init='k-means++', n_clusters=4, n_init=10 )
    # kmeans.fit( data )
    # print(list(kmeans.labels_))
    # print( list(labels ))
    # print( [ (a,b) for a,b in zip( kmeans.labels_, labels ) ] )
    # from collections import defaultdict
    # d = defaultdict(int)
    # for a,b in zip( kmeans.labels_, labels ):
    #     d[ (a,b) ] += 1
    
    # print( d )
    # exit(0)

    fit_times = []

    for metric in Metric:

        x, y = [], []
        for n_clusters in range( 2, N ):
            if args.clustering == 'kmeans':
                # t0 = time()
                kmeans = KMeans( init='k-means++', n_clusters=n_clusters, n_init=10 )
                t0 = time()
                kmeans.fit( data )
                fit_times.append( time() - t0 )
                # print( 'Fit time: %.2fs' % ( time() - t0 ) )
                x.append( n_clusters )
                y.append( get_kmeans_metric( metric, kmeans, data, labels ) )
            elif args.clustering == 'em':
                em = GaussianMixture( n_components = n_clusters, n_init = 10 )
                em.fit( data )
                y.append( get_em_metric( metric, em, data, labels ) )
            else:
                raise ValueError( 'Invalid clustering algorithm' )

        # x.append( n_clusters )
        # y.append( get_kmeans_metric( metric, kmeans, data, labels ) )

        plt.plot( [i for i in range(2,N)], y )
    plt.legend( [ m.name for m in Metric ] )
    plt.ylabel( 'Score' )
    plt.xlabel( 'Number of Clusters' )
    title = '%s - %s' % ( dataset.name, 'K Means' if args.clustering == 'kmeans' else 'Expectation Maximization' )
    plt.title( title )
    plt.show()
    print( 'Mean fit time: %.2f' % np.mean( fit_times ) )
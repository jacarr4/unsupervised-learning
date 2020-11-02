from data_loader import Dataset, get_dataset, best_params

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
from sklearn.feature_selection import VarianceThreshold
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
    # parser.add_argument( '--reducer', action = 'store', dest = 'reducer', required = True )
    args = parser.parse_args()

    dataset = Dataset[ args.dataset ]
    n_samples, n_features, n_labels, data, labels = get_dataset( dataset )

    print( 'Running on dataset %s' % dataset.name )

    for reducer in [ 'pca', 'ica', 'rp', 'vt' ]:
        n_components = best_params[ dataset ]
        if reducer == 'pca':
            reduced_data = PCA( n_components = n_components ).fit_transform( data )
            reducer_title = 'PCA'
        elif reducer == 'ica':
            reduced_data = FastICA( n_components = n_components ).fit_transform( data )
            reducer_title = 'ICA'
        elif reducer == 'rp':
            reduced_data = GaussianRandomProjection( n_components = n_components ).fit_transform( data )
            reducer_title = 'Random Projections'
        elif reducer == 'vt':
            reduced_data = VarianceThreshold( threshold = 1 ).fit_transform( data )
            reducer_title = 'Variance Threshold'
        else:
            raise ValueError( 'Invalid dimensionality reduction algorithm' )

        N = 20

        plt.xticks(range(2,N))

        print( 'Running reducer: %s' % reducer )

        fit_times = []

        for metric in Metric:

            x, y = [], []
            for n_clusters in range( 2, N ):
                if args.clustering == 'kmeans':
                    kmeans = KMeans( init='k-means++', n_clusters=n_clusters, n_init=10 )
                    t0 = time()
                    kmeans.fit( reduced_data )
                    fit_times.append( time() - t0 )
                    x.append( n_clusters )
                    y.append( get_kmeans_metric( metric, kmeans, reduced_data, labels ) )
                elif args.clustering == 'em':
                    em = GaussianMixture( n_components = n_clusters, n_init = 10 )
                    em.fit( reduced_data )
                    y.append( get_em_metric( metric, em, reduced_data, labels ) )
                else:
                    raise ValueError( 'Invalid clustering algorithm' )

            plt.plot( [i for i in range(2,N)], y )
        plt.legend( [ m.name for m in Metric ] )
        plt.ylabel( 'Score' )
        plt.xlabel( 'Number of Clusters' )
        title = '%s - %s with %s' % ( dataset.name, 'K Means' if args.clustering == 'kmeans' else 'Expectation Maximization', reducer_title )
        plt.title( title )
        plt.show()
        print( 'Mean fit time: %.2f' % np.mean( fit_times ) )
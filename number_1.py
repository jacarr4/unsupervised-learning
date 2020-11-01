from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from enum import IntEnum, auto
import argparse
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import scale

np.random.seed(42)

sample_size = 300

class Dataset( IntEnum ):
    Digits      = auto()
    PimaIndians = auto()

def get_dataset( dataset : Dataset ):
    if dataset == Dataset.Digits:
        X_digits, y_digits = load_digits(return_X_y=True)
        data = scale(X_digits)

        n_samples, n_features = data.shape
        n_digits = len(np.unique(y_digits))
        labels = y_digits
        return n_samples, n_features, n_digits, data, labels
    
    elif dataset == Dataset.PimaIndians:
        df = pd.read_csv('pima/pima-indians-diabetes.csv')
        df.columns = [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome' ]

        data = df[ [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ] ]
        target = df[ [ 'Outcome' ] ]

        data = scale(data)
        n_samples, n_features = data.shape
        n_labels = len(np.unique(target))
        labels = target.values.ravel()
        return n_samples, n_features, n_labels, data, labels


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

    for metric in Metric:

        x, y = [], []
        for n_clusters in range( 2, N ):
            if args.clustering == 'kmeans':
                kmeans = KMeans( init='k-means++', n_clusters=n_clusters, n_init=10 )
                kmeans.fit( data )
                x.append( n_clusters )
                y.append( get_kmeans_metric( metric, kmeans, data, labels ) )
            elif args.clustering == 'em':
                em = GaussianMixture( n_components = n_clusters, n_init = 10 )
                em.fit( data )
                y.append( get_em_metric( metric, em, data, labels ) )
            else:
                raise ValueError( 'Invalid clustering algorithm' )

        plt.plot( [i for i in range(2,N)], y )
    plt.legend( [ m.name for m in Metric ] )
    plt.ylabel( 'Score' )
    plt.xlabel( 'Number of Clusters' )
    title = '%s - %s' % ( dataset.name, 'K Means' if args.clustering == 'kmeans' else 'Expectation Maximization' )
    plt.title( title )
    plt.show()
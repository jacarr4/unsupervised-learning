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


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.
        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.
    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt
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
    print( df )

    fig, axes = plt.subplots( 3, 1, figsize = ( 10, 15 ) )
    plot_learning_curve( MLPClassifier(), 'Neural Network with Clustering', X, y, axes )
    plt.show()
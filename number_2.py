from data_loader import Dataset, get_dataset

import argparse
from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import scale

np.random.seed(42)

class DimReducer():
    def __init__( self, N, dataset, data ):
        self.N = N
        self.dataset = dataset
        self.data = data

        self.x = range( 2, self.N )
    
    def run_PCA( self ):
        y = []

        for n_components in range( 2, self.N ):
            pca = PCA( n_components = n_components )
            reduced_data = pca.fit_transform( self.data )
            y.append( np.sum( pca.explained_variance_ ) )

        plt.plot( self.x, y )
        plt.xlabel( 'Number of Components' )
        plt.ylabel( 'Explained Variance' )
        plt.title( '%s - %s' % ( self.dataset.name, 'PCA' ) )
        plt.show()

    def run_ICA( self ):
        avg_kurtoses = []

        for n_components in self.x:
            ica = FastICA(n_components = n_components)
            reduced_data = ica.fit_transform(self.data)

            k = kurtosis( reduced_data )
            m = np.mean( abs( k ) )
            avg_kurtoses.append( m )

        plt.plot( [i for i in self.x], avg_kurtoses )
        plt.xlabel( 'Number of Components' )
        plt.ylabel( 'Average Kurtosis' )
        plt.title( '%s - %s' % ( self.dataset.name, 'FastICA' ) )
        plt.show()

    def run_RP( self ):
        y = []

        for n_components in self.x:
            grp = GaussianRandomProjection( n_components = n_components )
            reduced_data = grp.fit_transform( self.data )
            print( reduced_data.shape )

    def run_VT( self ):
        y = []

        # for n_components in self.x:
        vt = VarianceThreshold( threshold = 1 )
        reduced_data = vt.fit_transform( self.data )
        reconstructed = vt.inverse_transform( reduced_data )
        print( reduced_data.shape )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--dataset', action = 'store', dest = 'dataset', required = True )
    args = parser.parse_args()
    
    dataset = Dataset[ args.dataset ]
    n_samples, n_features, n_labels, data, labels = get_dataset( dataset )

    dr = DimReducer( n_features + 1, dataset, data )
    # dr.run_PCA()
    # dr.run_ICA()
    dr.run_RP()
    # dr.run_VT()

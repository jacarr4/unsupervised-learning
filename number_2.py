from data_loader import Dataset, get_dataset

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
from sklearn.preprocessing import scale

np.random.seed(42)

X_digits, y_digits = load_digits(return_X_y=True)
data = scale(X_digits)

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

sample_size = 300

# df = pd.read_csv('pima/pima-indians-diabetes.csv')
# df.columns = [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome' ]

# data = df[ [ 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age' ] ]
# target = df[ [ 'Outcome' ] ]

# data = scale(data)
# n_samples, n_features = data.shape
# n_labels = len(np.unique(target))
# labels = target.values.ravel()

for n_components in range( 2, 20 ):
    reduced_data = FastICA(n_components = n_components, max_iter = 1000).fit_transform(data)

    # print( data )
    # print( reduced_data )

    print( kurtosis( reduced_data ) )
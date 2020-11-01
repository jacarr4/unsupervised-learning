from enum import auto, IntEnum
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np


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
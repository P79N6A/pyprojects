from sklearn.base import BaseEstimator
import numpy as np


class Never5Classifier(BaseEstimator):
    def __init__(self):
        pass

    def fit(self,X,y):
        pass

    def predict(self,X):
        return np.zeros((len(X),1),dtype=bool)
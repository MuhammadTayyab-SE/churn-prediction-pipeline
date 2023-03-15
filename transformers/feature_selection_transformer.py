import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold) -> None:
        super().__init__()
        self.threshold = threshold
        self.filtered_features = []
        print(">>>>>> feature transformer constructor called")

    
    def fit(self, X, y):
        print(">>>>>> feature transformer fit method called")
        target_column = y.name
        df = X.copy()
        df.reset_index(inplace=True, drop=True)
        y.reset_index(inplace=True, drop=True)
        y_df = pd.DataFrame(y)
        df = pd.concat([df, y_df],axis=1)
        for column, value in df.corr()[target_column].items():
            if abs(value) * 100 > self.threshold and column not in self.filtered_features:
                self.filtered_features.append(column)
        if target_column in self.filtered_features:
            self.filtered_features.remove(target_column)
        return self
    
    def transform(self, X, y=None):
        print(">>>>>> feature transformer transform called")
        lst = self.filtered_features    
        return X[lst]
    
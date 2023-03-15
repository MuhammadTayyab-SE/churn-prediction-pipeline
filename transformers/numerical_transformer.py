import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, target_column="Attrition"):
        self.scaler = StandardScaler()
        self.target_column = target_column
        print(">>>>> NumericalTransformer constructor called")
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X, y=None):
        print(">>>>> Numerical Transformer transform method called")
        # normalize the dataframe
        X.reset_index(inplace=True, drop=True)
        original_columns = X.columns
        if self.target_column in X.columns:
            df_labels = X[self.target_column]
            X.drop(self.target_column, axis=1, inplace=True)
        columns = X.columns
        
        # normalize the dataframe
        df_normalized = self.scaler.fit_transform(X)
        
        # create a new dataframe with the normalized values
        df_normalized = pd.DataFrame(df_normalized, columns=columns)
        
        if self.target_column in original_columns:
            df_normalized = pd.concat([df_normalized, df_labels], axis=1)
    
        return df_normalized
    
    def fit_transform(self, X, y=None, **fit_params):
        return super().fit_transform(X, y, **fit_params)
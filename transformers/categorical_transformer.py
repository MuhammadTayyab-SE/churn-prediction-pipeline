import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    This class encodes categorical features into Categorical variables
    """
    
    def __init__(self, drop_single_column=True):
        self.drop_single_column =drop_single_column
        self.feature_encoding_dictionary = {}
        self.y_stored = pd.Series()
        self.feature_columns = []
        print("CategoricalTransformer constructor is called")
    
    def fit(self,X, y):
        print(">>>>>> Categorically fit called")

        if self.drop_single_column:
            X = self.drop_single_value_columns(X)
            
        # features encoding
        object_cols = self.get_object_cols(X)
        self.feature_encoding_dictionary = self.generate_feature_encoding_dictionary(X, object_cols)
        # X = self.encode_data(X, object_cols, self.feature_encoding_dictionary)
        
        self.feature_columns = X.columns
        return self

    def transform(self, X, y=None):
        print(">>>>>> Categorically transformer called")
        
        X = X[self.feature_columns]
            
        object_cols = self.get_object_cols(X)
        X = self.encode_data(X,object_cols, self.feature_encoding_dictionary)
        X.reset_index(inplace=True, drop=True)
        df_transformed = X
            
        return df_transformed
    
    
    def fit_transform(self, X, y=None, **fit_params):
        print(">>>>>> Categorically fit_transformer called")
        self.fit(X,y)
        return self.transform(X,y)

    def get_object_cols(self, X):
        object_cols = [col for col, dtype in X.dtypes.items() if dtype == "object" or dtype == "string"]
        return object_cols
    
    def drop_single_value_columns(self, df):
        """Drop those columns that contain same values"""
        for column in df.columns:
            if len(df[column].unique()) < 2:
                df.drop(column,axis=1 ,inplace=True)
        return df

    def generate_feature_encoding_dictionary(self, df, columns):
        """
        Generate encoding dictionary that contains mapping of column names and categorical variables
        """
        integer = 0
        encoding = {}
        encoding_dictionary = {}
        for column in columns:
            for category in sorted(df[column].unique()):
                encoding[category] = integer
                integer += 1
            encoding_dictionary[column]  = encoding
            integer = 0
            encoding = {}
        return encoding_dictionary
    
    def encode_data(self, df, columns, encoding_dictionary):

        """Convert non-categorical column into categorical column """
        if len(columns) == 1:
            df = df.apply(lambda x: encoding_dictionary[x])
        else:
            for column in columns:
                dictionary = encoding_dictionary[column]
                df[column] = df[column].apply(lambda x: dictionary[x])
        return df
    
    def encode_column(self,series):
        encoded_series = self.encode_data(series, [series.name], self.label_encoding_dictionary)
        return encoded_series
        


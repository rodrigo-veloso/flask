from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import Isomap
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
import keras
import umap
import numpy as np
import pandas as pd

class Autoencoder:
    
    def __init__(self, n_components, n_layers = 1, **kwargs):
        self.n_components = n_components
        self.n_layers = n_layers
        self.kwargs = kwargs

    def fit(self, X, y = None):
        #if isinstance(X,pd.DataFrame):
        #    X = X.values
        #print(X)
        input_ = keras.layers.Input(shape=(X.shape[1]))
        encoded = keras.layers.Dense(self.n_components, activation='relu')(input_)
        decoded = keras.layers.Dense(X.shape[1], activation='relu')(encoded)

        self.autoencoder = keras.Model(input_,decoded)
        self.encoder = keras.Model(input_, encoded)
        self.autoencoder.compile(loss = keras.losses.MeanSquaredError())
        print(X.shape[1])
        self.autoencoder.fit(X, X, epochs = 100, batch_size = 64, shuffle=True)

    def transform(self, X, y = None):
        return self.encoder.predict(X)

class DimensionalityReducer:
    
    def __init__(self, reducer, **kwargs):
        """
        Constructor
        
    	Parameters
    	----------            
        selector : str
                   name of algorithm to be applied
        **kwargs : 
                   optional and positional arguments of the choosen algorithm (selector)
    	Returns
    	-------
        FeatureSelector
      Examples
      ---------
      variance thresholding:      f = FeatureSelector('variance', threshold=0.3) #Instantiating 
                                  f.fit(X[,y]) #fitting (y is optional for variance thresholding)
                                  X = f.transform(X) #transforming

      filter-based, k best (MAD): f = FeatureSelector('univariate_kbest', score_func=FeatureSelector.mean_abs_diff, k=2) #Instantiating 
                                  #score_func can be any function f: R^n -> R^n (n = number of columns)
                                  f.fit(X,y) #fitting 
                                  X = f.transform(X) #transforming

      wrapper, recursive:         f = FeatureSelector('recursive', estimator = LinearSVC(), n_features_to_select=2) #Instantiating 
                                  #estimator should be an instance of a classification or regression model class from scikit-learn 
                                  #one can use a custom class but it must be compatible with scikit-learn arquitecture  
                                  f.fit(X,y) #fitting 
                                  X = f.transform(X) #transforming

     wrapper, sequential:          f = FeatureSelector('sequential', estimator = LinearSVC(), direction='forward') #Instantiating 
                                  #estimator should be an instance of a classification or regression model class from scikit-learn 
                                  #one can use a custom class but it must be compatible with scikit-learn arquitecture  
                                  f.fit(X,y) #fitting 
                                  X = f.transform(X) #transforming
   
      to better understand the optional arguments of each algorithm see: https://scikit-learn.org/stable/modules/feature_selection.html                         
        """
        self.reducer = reducer
        self.reducers = {'factor_analysis': FactorAnalysis, 
                         'pca': PCA,
                         'ica': FastICA,
                         'isomap': Isomap,
                         'locally_linear_embedding': LocallyLinearEmbedding,
                         'spectral_embedding': SpectralEmbedding,
                         'tsne': TSNE,
                         'mds':MDS,
                         'umap':umap.UMAP,
                         'autoencoder':Autoencoder}
        self.kwargs = kwargs 
        self.fitted = False
        self.reduction = self.reducers[self.reducer](**self.kwargs)
        
    
    def fit(self, X: pd.DataFrame, y = None):
        """
        Identify the features to be selected.
        
    	Parameters
    	----------            
        X : pd.DataFrame
             features to be selected

        y : pd.DataFrame
            target values
                    
    	Returns
    	-------
        None
        """
        self.columns = X.columns
        self.reduction.fit(X,y)
        self.fitted = True

    def transform(self, df: pd.DataFrame):
        """
        Select features based on fit
        
    	Parameters
    	----------            
        pd.DataFrame
        dataframe with features to be selected
                    
    	Returns
    	-------
        df : pd.DataFrame
             dataframe with selected features only
        """
        if not self.fitted:
            raise Exception("Not yet trained.")

        
        #return self.selection.transform(df)
        return self.reduction.transform(df)

    def fit_transform(self, df: pd.DataFrame):
        """
        Select features based on fit
        
    	Parameters
    	----------            
        pd.DataFrame
        dataframe with features to be selected
                    
    	Returns
    	-------
        df : pd.DataFrame
             dataframe with selected features only
        """

        
        #return self.selection.transform(df)
        return self.reduction.fit_transform(df)
    
    def inverse_transform(self, df: pd.DataFrame):
        """
        Apply the invese_transform of vectorizer to each column
        Options: index, bag_of_words and tf_idf
        
    	Parameters
    	----------            
        df : pd.DataFrame
             dataframe with columns to be unvectorizer
                    
    	Returns
    	-------
        pd.DataFrame
        """
        if not self.fitted:
            raise Exception("Not yet trained.")

        
        #return self.selection.transform(df)
        return self.reduction.inverse_transform(df)

    @staticmethod
    def mean_abs_diff(X, y=None):
        """
        method to compute the mean absolute difference (MAD) of all atributes of X
        
    	Parameters
    	----------            
        X : pd.DataFrame
             dataframe 
        y: any type
            not necessary, used only for compatibility issues
                    
    	Returns
    	-------
        pd.DataFrame
        """
        return np.sum(np.abs(X - np.mean(X, axis = 0)), axis = 0)/X.shape[0]

    @staticmethod
    def variance(X, y=None):
        """
        method to compute the mean variance of all atributes of X
        
    	Parameters
    	----------            
        X : pd.DataFrame
             dataframe 
        y: any type
            not necessary, used only for compatibility issues
                    
    	Returns
    	-------
        pd.DataFrame
        """
        return np.sum((X - np.mean(X, axis = 0)**2), axis = 0)/X.shape[0]
  
    @staticmethod
    def disp_ratio(X, y=None):
        """
        method to compute the dispersion ratio of all atributes od X
        
    	Parameters
    	----------            
        X : pd.DataFrame
             dataframe
        y: any type
            not necessary, used only for compatibility issues
                    
    	Returns
    	-------
        pd.DataFrame
        """
        return np.mean(X, axis = 0)/np.power(np.prod(X, axis = 0),1/X.shape[0])

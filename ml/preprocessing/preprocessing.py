import pandas as pd

from ml.preprocessing.normalization import Normalizer
import logging

logging.getLogger().setLevel(logging.INFO)

class Preprocessing:
    """
    Class to perform data preprocessing before training
    """    
    def __init__(self, normalizer_dic = None):
      self.processes = []
      if normalizer_dic == None:
        self.normalizer = None
      else:
        self.normalizer = Normalizer(normalizer_dic)

    def clean_data(self, df: pd.DataFrame, append = True, **kwargs):
        """
        Perform data cleansing.
        
        Parameters
        ----------            
        df  :   pd.Dataframe
                Dataframe to be processed

        append  :   boolean
                    if clean_data should be added to processes

        Returns
    	-------
        pd.Dataframe
            Cleaned Data Frame
        """
        logging.info("Cleaning data")
        if append:
          self.processes.append([self.clean_data, kwargs])
        return df.dropna()

    def categ_encoding(self, df: pd.DataFrame, append = True, **kwargs):
        """
        Perform encoding of the categorical variables

        Parameters
        ----------            
        df  :   pd.Dataframe
                Dataframe to be processed

        append  :   boolean
                    if categ_encoding should be added to processes
        
        encoder: 
                 encoding method, if None use 

        columns: list
                 list of columns to be encoded, if None all columns are encoded

        Returns
    	-------
        pd.Dataframe
            Cleaned Data Frame
        """
        logging.info("Category encoding")

        encoder = kwargs.get('encoder')
        columns = kwargs.get('columns')
    
        if encoder:
          encoder=encoder(cols=columns,verbose=False,)
          if append:
            self.processes.append([self.categ_encoding, kwargs])
          return encoder.fit_transform(df)
        else:
          return pd.get_dummies(df)

    def apply_all(self, df):

      for process in self.processes:
        df = process[0](df,False,**process[1])
      if self.normalizer != None:
        df = self.normalizer.transform(df)
      return df

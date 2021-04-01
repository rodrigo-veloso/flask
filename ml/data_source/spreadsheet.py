import pandas as pd

from ml.data_source.base import DataSource

class Spreadsheet(DataSource):
    """
    Class to read files from spreadsheets or raw text files
    """
    
    def get_data(self, path, columns = None)->pd.DataFrame:
        """
        Returns a flat table in Dataframe
        
        Parameters
        ----------            
        arg : type
              description
  
        columns : list 
                  selected columns, if None returns all columns
        
        Returns
        -------
        pd.DataFrame
            Dataframe with data
        """
        if columns == None:
          return pd.read_csv(path)
        else:
          return pd.read_csv(path)[columns]

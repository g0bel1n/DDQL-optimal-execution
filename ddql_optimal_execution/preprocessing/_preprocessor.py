import numpy as np
import pandas as pd


def normalize(df: pd.Series) -> pd.Series:
    '''This function normalizes the price data to have a mean of 0 and a standard deviation of 1 after substracting 
    the first price value.
    
    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing financial data with a "Date" column and a "Price" column.
    
    Returns
    -------
        a pandas DataFrame.
    
    '''
    df -= df.iloc[0]
    df -= df.mean()
    df /= df.std()
    return df

class Preprocessor:
    '''
    This class is used to preprocess the data before it is fed into the environment. It splits the data into periods
    and optionally calculates the QV and normalizes the price.
    
    Attributes
    ----------
    n_periods : int
        an integer representing the number of periods in the time series data. It is the number of trading actions agents can take.
        QV : bool
        QV stands for "Quadratic Variation" and is a measure of volatility. This parameter is a boolean
        value that determines whether the QV should be calculated or not. If set to True, the QV will be
        calculated and added to the DataFrame.
        
        normalize_price : bool
        A boolean parameter that determines whether the price data should be normalized or not. If set to
        True, the price data will be normalized to have a mean of 0 and a standard deviation of 1 after substracting
        the first price value.
        
        Methods
        -------
        __call__(df)
            This function splits a pandas DataFrame into periods based on a specified number of periods, and
            optionally calculates the QV and normalizes the price.
                
    '''


    def __init__(self, n_periods : int, QV :bool = True, normalize_price : bool = True  ) -> None:
        '''This is a constructor function that initializes the object with the given parameters.
        
        Parameters
        ----------
        n_periods : int
            an integer representing the number of periods in the time series data. It is the number of trading actions agents can take.
        QV : bool, optional
            QV stands for "Quadratic Variation" and is a measure of volatility. This parameter is a boolean
        value that determines whether the QV should be calculated or not. If set to True, the QV will be
        calculated and added to the DataFrame.
        normalize_price : bool, optional
            A boolean parameter that determines whether the price data should be normalized or not. If set to
        True, the price data will be normalized to have a mean of 0 and a standard deviation of 1 after substracting 
        the first price value.
        
        '''
        self.n_periods = n_periods
        self.QV = QV
        self.normalize_price = normalize_price

    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        '''This function splits a pandas DataFrame into periods based on a specified number of periods, and
        optionally calculates the QV and normalizes the price.
        
        Parameters
        ----------
        df : pd.DataFrame
            A pandas DataFrame containing financial data with a "Date" column and a "Price" column.
        
        Returns
        -------
            a pandas DataFrame.
        
        '''
        df = df.set_index("Date")
        _date_splits = np.split(df.index, self.n_periods)
        df["period"] = 0
        df["period"] = df["period"].astype(int)
        for i, split in enumerate(_date_splits):
            df.loc[split, "period"] = i

        if self.QV:
            df["QV"] = df.groupby("period")["Price"].transform(lambda x: ((x - x.shift(1))**2).sum())
            df["QV"] -= df["QV"].mean()
            df["QV"] /= 2*df["QV"].std()

        if self.normalize_price:
            df["Price"] = normalize(df["Price"])

        for col in df.columns:
            df[col] = df[col].astype(float)
            df[col] = normalize(df[col])


        df = df.iloc[1:]

        return df
        
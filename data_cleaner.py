# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:26:36 2020

@author: Mat
"""

#goal is to create a genearl checker for missing data, and misentered data
#that can be imported for future projects

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame([range(3), range(3), [0,1, 1000000000], range(3), range(3)])


#assume data is the given pandas dataframe

"""Operations for missing data"""

#return which rows are missing data 
def find_missing_row_data(data):
    return data.isnull().any(axis = 1)


#return a new data frame without rows that are missing data
def delete_missing_row_data(data):
    rows = data.notna().all(axis = 1)
    return data[rows]

#return data frame only containing rows with missing data
def delete_full_row_data(data):
    return data[find_missing_row_data(data)]
    
#return any column with missing data
def find_missing_column_data(data):
    return data.isnull().any(axis = 0)

#return a new data frame without columns that are missing data
def delete_missing_column_data(data):
    columns = data.notna().all(axis = 0)
    return data.loc[:, columns]
    
#return a data frame with only columns that are missing data
def delete_full_column_data(data):
    return data.loc[:,find_missing_column_data(data)]
    
"""Operations for misentered data"""

#return columns with data of different types 
def find_mixed_column_data(data):
    return data.dtypes == object

#delete columns with data of mixed type
def delete_mixed_column_data(data):
    return data.loc[:, data.dtypes != object]

#return columns with mixed data types 
def mixed_columns(data):
    return data.loc[:, find_mixed_column_data(data)]

#find extreme values on numeric rows only 
#limit is the number of allow standard deviations away from mean
def find_extreme_values(data, limit):
    df = data.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    scaler.fit(df)
    df = scaler.transform(df)
    df = pd.DataFrame(df)
    return df.abs().max(axis = 1) >= limit

#delete rows with extreme values defined by number of standard deviations away from mean

def delete_extreme_values(data, limit):
    return data.loc[~find_extreme_values(data, limit), :]  

"""Create a clean data set by deleting all missing and misentered data"""

#delete all bad data
def simple_clean(data,limit):
    df = data
    df = delete_missing_row_data(df)
    df = delete_mixed_column_data(df)
    df = delete_extreme_values(data, limit)
    return df


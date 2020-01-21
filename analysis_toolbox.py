"Several functions to more easily explore trends in data"

import pandas as pd
import numpy as np

#separate training set using random sampling
from sklearn.model_selection import train_test_split

def seperate_data(data, size):
    train_set, test_set = train_test_split(data, test_size= size, random_state=42)
    return [train_set,test_set]

#seperate data using stratfied sampling 
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_sample(data, size, target_column):
    split = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=42)
    for train_index, test_index in split.split(data, data[target_column]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    
    for set in (strat_train_set, strat_test_set):
        set.drop([target_column], axis=1, inplace=True)
        
    return [strat_train_set, strat_test_set]
        
#use correlation to observe trends
def correlation_matrix(data, column):
    corr_matrix = data.corr()
    corr_matrix[column].sort_values(ascending = False)
    print(corr_matrix)
    return(corr_matrix)

#create plots for multiple attribures, attributes should be in a list 
from pandas.plotting import scatter_matrix

def scatter_plot_matrix(data, attributes, width, height):
    scatter_matrix(data[attributes], figsize = (width, height))
    

#separate categorial variable into binary variables
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def cat_to_binary(data, column):
    df_cat = data[column]
    encoder = LabelEncoder()
    df_encoded = encoder.fit_transform(df_cat)
    encoder = OneHotEncoder(categories='auto')
    df_cat = encoder.fit_transform(df_encoded.reshape(-1,1))
    df_cat = df_cat.toarray()
    df_cat = pd.DataFrame(df_cat)
    data = data.join(df_cat)
    del data['Pclass']
    return data

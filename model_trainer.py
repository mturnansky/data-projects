# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:18:39 2020

@author: Mat
"""

"""Train data on several classic models assuming data. Before using this one 
should have already cleaned, separated training set, and used some cursory
exploration tools."""

import pandas as pd
import numpy as np


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RaondmForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LinearRegression

#run cross evaluation on several common machine learning algorithms
def Tester(data, target, cross_sections, depth):
    #decision tree
    tree = DecisionTreeClassifier(random_state=0)
    tree_scores = pd.Series(cross_val_score(tree, data, target, cv= cross_sections))
    
    #random forest
    forest =  RaondmForestClassifier(n_estimators = 100)
    forest_scores = pd.Series(cross_val_score(forest, data, target, cv = cross_sections))
    
    #stochastic gradient descent
    sgd = SGDClassifier()
    sgd_scores = pd.Series(cross_val_score(sgd, data, target, cv = cross_sections))  
    
    #lienar regression
    regression =  LinearRegression()
    regression_scores = pd.Series(cross_val_score(regression, data, target, cv = cross_sections))

    #return all cross evaluations in a new dataframe 
    test_scores = pd.DataFrame({'Decision Tree': tree_scores, 'Random Forest': forest_scores, \
                            'SGD': sgd_scores, 'Regression': regression_scores})
    
    return test_scores


#random forest 
def Random_Forest(data, target, depth):
    rf = RaondmForestClassifier(max_depth = depth, random_state = 0)
    train = rf.fit(data, target)
    return train.predict(data)




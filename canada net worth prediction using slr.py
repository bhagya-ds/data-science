# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:16:15 2020

@author: hp lap
"""
#  predicting  the canada net income  worth of 2020 AND 2021

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#impport the dataset
data=pd.read_csv("C:/Users/hp lap/Desktop/canada_income.csv")
data

X=data.iloc[:,:-1].values
y=data.iloc[:,1].values

#spilting the data in training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_train)
pd.DataFrame(X_train,y_pred)
# finding the model accuracy 
regressor.score(X_test,y_test) # i got 75%
# visulization using scatter plot for training data
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('year vs canada net income')
plt.xlabel('Year')
plt.ylabel('net income')
plt.show()

#predict the new value (example )
regressor.predict([[2021]])  #43359.45
regressor.predict([[2020]])   #42511.94


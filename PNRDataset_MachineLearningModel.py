# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 23:01:49 2020

@author: msigl6595dk
"""

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv(r'trips_reduced.csv', header=0, sep=',', quotechar='"')
dataset.info()

X=dataset.loc[:, ['CabinCategory','CreationDate','DepartureTime','nPAX']].values
y=dataset.iloc[:,-1].values

#LabelEncoding Of dependent Variable

le=LabelEncoder()
y=le.fit_transform(y)

#split training and test data sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)

#Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train)

X_test = sc.fit_transform(X_test)
print(X_test)

#Training LinearRegression Model with Training data
regressor = LinearRegression()
regressor.fit(X_train,y_train)
#Testing the Model with test data
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#Predictions vs test data
Model_results=np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)
# creating a list of column names
column_values = ['Predicted_Results', 'Test_Results']
# creating the dataframe
Model_Output = pd.DataFrame(data = Model_results,columns = column_values)
#Storing results to CSV
Model_Output.to_csv(r'trips_output.csv', index = False)


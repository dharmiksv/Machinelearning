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
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

dataset = pd.read_csv(r'trips_reduced.csv', header=0, sep=',', quotechar='"')
dataset.info()

X=dataset.loc[:, ['ArrivalTime','CabinCategory','CreationDate','CurrencyCode','DepartureTime','Destination','OfficeIdCountry','Origin','TotalAmount','nPAX']].values

y=dataset.iloc[:,-1].values

#LabelEncoding Of dependent Variable

le=LabelEncoder()
y=le.fit_transform(y)

#Label encoding the CurrencyCode(3),Destination(5),OfficeIdCountry(6),Origin(7)
X[:,3]=le.fit_transform(X[:,3])
X[:,5]=le.fit_transform(X[:,5])
X[:,6]=le.fit_transform(X[:,6])
X[:,7]=le.fit_transform(X[:,7])

#split training and test data sets

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=1)


#Create dataframe from data in X_train
#label the colums using the strings in input dataset

features_train_df=pd.DataFrame(X_train,columns=['ArrivalTime','CabinCategory','CreationDate','CurrencyCode','DepartureTime','Destination','OfficeIdCountry','Origin','TotalAmount','nPAX'])

# create a scatter matrix from the dataframe, color by y_train

#smat=pd.plotting.scatter_matrix(features_train_df,alpha=0.2)
#,c=y_train,figsize=(15,15),marker='o',hist_kwds={'bins':20},s=60,
#Feature Scaling

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
print(X_train)


X_test = sc.fit_transform(X_test)
print(X_test)

#Training LinearRegression Model with Training data

regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Evaluting the Model with test data
y_pred = regressor.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
np.set_printoptions(precision=2)

#Test set score: 0.24
print("Test set score: {:.2f}".format(regressor.score(X_test,y_test)))

#print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
#Predictions vs test data
Model_results=np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1)
# creating a list of column names
column_values = ['Predicted_Results', 'Test_Results']
# creating the dataframe
Model_Output = pd.DataFrame(data = Model_results,columns = column_values)
#Storing results to CSV
Model_Output.to_csv(r'trips_output.csv', index = False)



#Training K-Nearest Neighbors Model with Training data
#Test Score - 0.79

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)

#Evaluting the Model with test data
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
np.set_printoptions(precision=2)

#Test set score
print("Test set score: {:.2f}".format(knn.score(X_test,y_test)))


#Training Logistic Regression Model with Training data
#Test Score - 0.79

lr = LogisticRegression()
lr.fit(X_train,y_train)

#Evaluting the Model with test data
y_pred = lr.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
np.set_printoptions(precision=2)

#Test set score
print("Test set score: {:.2f}".format(lr.score(X_test,y_test)))

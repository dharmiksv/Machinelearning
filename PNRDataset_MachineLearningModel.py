import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectPercentile
import csv


dataset = pd.read_csv(r'trips_reduced_2.csv', header=0, sep=',', quotechar='"')
df=dataset
"""
ArrivalTime – local time of arrival
BusinessLeisure – if the trip is for business or leisure
CabinCategory – cabin class
CreationDate –PNR creation date (Julian day)
CurrencyCode – 3-letter currency code of payment
DepartureTime – local time of departure
Destination – IATA code of arrival airport
OfficeIdCountry – country code of office placing the reservation
Origin – IATA code of departure airport
TotalAmount – total reservation cost
nPAX – number of passengers

"""


print('Dataframe dimensions:', df.shape)
#____________________________________________________________
# gives some infos on columns types and number of null values
tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))
tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)
                         .T.rename(index={0:'null values (%)'}))
tab_info



X=dataset.loc[:, ['CreationDate','DepartureTime']].values   #processing only 2 features /10 features
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


#Training Logistic Regression Model with Training data

clf = LogisticRegression(C=1e5)

# Fit the classifier
clf.fit(X_train,y_train)

#Evaluting the Model with test data
y_pred = clf.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
np.set_printoptions(precision=2)

#Test set score
print("Test set score: {:.2f}".format(clf.score(X_test,y_test)))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
# Plot also the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Creation Date')
plt.ylabel('Departure Time')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks((range(-3,3)))
plt.yticks((range(-3,3)))
plt.show()



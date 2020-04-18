#Multiple Linear Regression


#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values


# Importing categorical Data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# TO ENCODE 
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)


#Avoiding the Dummy Variable Trap
X = X[:,1:]


#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


#Fitting Simple Linear Regression to the training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


#Predicting the Test set results
# Create Vector of Predictions
Y_pred = regressor.predict(X_test)
 #regressor.predict([[6.5]])




# Building the optimal model using Backward Elimination
import statsmodels.api as sm
# To add all ones at end X = np.append(arr = X, values = np.ones((50,1)).astype(int),axis = 1) 
X = np.append(arr = np.ones((50,1)).astype(int), values = X,axis = 1) 
X_opt = X[:,[0,1,2,3,4,5]]
#OLS --->Ordinary least Square
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Higher P value is taken out from P val
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()






























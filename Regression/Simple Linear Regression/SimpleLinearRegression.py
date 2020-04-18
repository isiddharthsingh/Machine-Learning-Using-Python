# Simple linear Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values


#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 1/3, random_state = 0)


#Fitting Simple Linear Regression to the training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)


#Predicting the Test set results
# Create Vector of Predictions
Y_pred = regressor.predict(X_test)


# Visualising the Training set results
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Salary VS Experience (Training Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()


# Visualising the Test set results
plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.title('Salary VS Experience (Test Set)')
plt.xlabel('Years Of Experience')
plt.ylabel('Salary')
plt.show()
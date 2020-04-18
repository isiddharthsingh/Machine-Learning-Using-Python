#Polynomial Regression


#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#No splitting dataset into training and test set as the data is very small
#No feature scaling either as we are use same Linear Regression Class


#Fitting Linear Regression to the dataset just for comparison with Polynomial
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
#Now to get X**2 term of X we change the degree
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)


# Visualising the Linear Regression Model
plt.scatter(X,y,color = 'Red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show



#Visualising the Polynomial Regression Model
#To get a more curve in plot use next two lines 
X_grid = np.arange(min(X),max(X),0.1) #This gives us a vector
X_grid = X_grid.reshape((len(X_grid),1)) # To convert it back to matrix
plt.scatter(X,y,color = 'Red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show


#Predicting a new result with Linear Regression
lin_reg.predict([[6.5]])


#Predicting a new result with Polynomial Regression
lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
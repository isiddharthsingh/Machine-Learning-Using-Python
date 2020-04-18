
#Polynomial Regression


#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)


#Regressor




#Predicting a new result with Linear Regression
regressor.predict([[6.5]])


# Visualising the  Regression Model
plt.scatter(X,y,color = 'Red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show



#Visualising the  Regression Model in higher Quality
#To get a more curve in plot use next two lines 
X_grid = np.arange(min(X),max(X),0.1) #This gives us a vector
X_grid = X_grid.reshape((len(X_grid),1)) # To convert it back to matrix
plt.scatter(X,y,color = 'Red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show




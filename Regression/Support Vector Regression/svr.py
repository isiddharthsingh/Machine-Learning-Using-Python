
#Support Vector Regression


#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(np.array([y]).reshape(-1,1))


#Fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf',gamma='auto')
regressor.fit(X,y)




#Predicting a new result with Linear Regression
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))


# Visualising the SVR results
plt.scatter(X,y,color = 'Red')
plt.plot(X,regressor.predict(X),color = 'blue')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show



#Visualising the  Regression Model in higher Quality
#To get a more curve in plot use next two lines 
X_grid = np.arange(min(X),max(X),0.1) #This gives us a vector
X_grid = X_grid.reshape((len(X_grid),1)) # To convert it back to matrix
plt.scatter(X,y,color = 'Red')
plt.plot(X_grid,regressor.predict(X_grid),color = 'blue')
plt.title('Truth or Bluff(SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show




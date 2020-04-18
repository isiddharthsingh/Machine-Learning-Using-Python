# Artificial Neural Network

#Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#Encoding Categorical Data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1,2])], remainder="passthrough")
ct_country_gender = np.array(ct.fit_transform(X)[:, [1,2,3]], dtype=np.float)
X = np.hstack((ct_country_gender[:, :2], dataset.iloc[:, 3:4].values, ct_country_gender[:, [2]], dataset.iloc[:, 6:-1].values))




# Splitting the dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)



# feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#y = sc_y.fit_transform(np.array([y]).reshape(-1,1))


# Importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense



# Initialising the ANN
classifier = Sequential()


#Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6,kernel_initializer=  'uniform' , activation = 'relu', input_dim = 11))

#Adding second hidden layer
classifier.add(Dense(units = 6,kernel_initializer=  'uniform' , activation = 'relu'))

#Adding the Output Layer
classifier.add(Dense(units = 1,kernel_initializer=  'uniform' , activation = 'sigmoid'))


#Compiling ANN 
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


#Fitting the ANN to training Set
classifier.fit(X_train,y_train,batch_size=10,epochs=100)


# Predicting the Test Set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

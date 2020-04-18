# Data Preprocessing

# Importing Librariies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the Dataset
dataset = pd.read_csv('Data.csv')
#Creating the matrix of features
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:,3].values



# Take care of missing Data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
# To fit the Imputer object in matrix X
imputer=imputer.fit(X[:,1:3])
X[:,1:3]= imputer.transform(X[:,1:3])


# Importing categorical Data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# TO ENCODE ON THE FIRST COLUMN THAT IS COUNTRY[0]
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Encoding Y data
#since it is the dependent variable, we're going to use only label encoder
from sklearn.preprocessing import LabelEncoder
Y = LabelEncoder().fit_transform(Y)


#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size = 0.2,random_state = 0)


# feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# for train set we have to fit and trasform 
#for test we have to just transform
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



























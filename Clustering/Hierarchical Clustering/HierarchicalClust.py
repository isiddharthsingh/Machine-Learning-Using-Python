# Hierarchical Clusterning


#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:,[3,4]].values


# Using dendrogram to find optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidian Distanes')
plt.show()


# Fitting hierachica clustering to mall dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5 ,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)


#Visualising the Clusters
plt.scatter(X[y_hc== 0,0],X[y_hc == 0,1],s = 100,c = 'red',label= 'Careful')
plt.scatter(X[y_hc== 1,0],X[y_hc == 1,1],s = 100,c = 'blue',label= 'Standard')
plt.scatter(X[y_hc== 2,0],X[y_hc == 2,1],s = 100,c = 'pink',label= 'Target')
plt.scatter(X[y_hc== 3,0],X[y_hc == 3,1],s = 100,c = 'magenta',label= 'Careless')
plt.scatter(X[y_hc== 4,0],X[y_hc == 4,1],s = 100,c = 'yellow',label= 'Sensible')
plt.title("Cluster of Clients")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()
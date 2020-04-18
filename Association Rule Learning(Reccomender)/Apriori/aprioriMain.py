# Apriori


#Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header= None)


transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range (0,20)])


#Training Apriori on the Dataset
from apyori import apriori
#min_support = 3*7/7500  min 3 times an object a day 
rules = apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_lenght=2)


#Visualising the results
result = list(rules)

'''#Now to show the relations
#if upper code does not show properly
results_list = []

for i in range(0, len(result)):

    results_list.append('RULE:\t' + str(result[i][0]) + '\nSUPPORT:\t' + str(result[i][1])+'\nConfidence:\t' + str(result[i][2]))'''
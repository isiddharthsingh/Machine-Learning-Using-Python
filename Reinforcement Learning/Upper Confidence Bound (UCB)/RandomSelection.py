# Random Selection

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')


# Implementing Random Selection
import random
N= 10000
d = 10
ads_selected = []
total_rewards = 0
for n in range(0,N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    total_rewards += reward
    
    
    
# Visualising the Results -Histogram
plt.hist(ads_selected,rwidth=0.25)
plt.title('Histogram of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected') 
plt.show()   
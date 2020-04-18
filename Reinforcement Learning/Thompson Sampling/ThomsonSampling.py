# Thomson Sampling


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')



#Implementing Thompson Sampling
#check image in folder for Formulas
import random
N=10000        #No, of users
d=10           #No. of ADS
numbers_of_rewards_1 = [0]*d
numbers_of_rewards_0 = [0]*d
ads_selected = []
total_reward = 0
for n in range(0,N):
    ad = 0
    max_random_draw = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        
        if (random_beta>max_random_draw):
            max_random_draw = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if(reward == 1):
        numbers_of_rewards_1[ad] +=1
    else:
        numbers_of_rewards_0[ad] +=1
    total_reward = total_reward + reward
    
    

#Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of Ads Selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected') 
plt.show()   
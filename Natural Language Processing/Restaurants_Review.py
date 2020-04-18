# NLP


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#Importing the Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv' , delimiter = '\t', quoting = 3)


# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ',dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

''' To clean HTML tags
import re
 
def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext
'''
    

#Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values


# Splitting the dataset into test set and training set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20, random_state = 0)



#Fitting Naive Bayes to Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)


# Predicting the Test Set results
y_pred = classifier.predict(X_test)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)



'''
We can use the following function to predict a new review.

def predict(new_review):
    new_review = re.sub("[^a-zA-Z]", " ", new_review)
    new_review = new_review.lower().split()
    new_review = [ps.stem(word) for word in new_review if word not in set(stopwords.words("english"))]
    new_review = " ".join(new_review)
    new_review = [new_review]
    new_review = cv.transform(new_review).toarray()
if classifier.predict(new_review)[0] == 1:
    return "Positive"
else:
    return "Negative"
'''
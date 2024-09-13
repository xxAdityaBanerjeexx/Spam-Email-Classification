# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:40:36 2024

@author: Asit Banerjee
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score

#Loading data
data=pd.read_csv("C:/Users/Asit Banerjee/spam.csv")
x=data.Message
y=data.Category


#One-Hot Encoding of Nominal Data
for i in range(len(y)):
    if y[i] == 'spam':
        y[i]=0
    else:
        y[i]=1


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

#Feature Extraction
feat_ext= TfidfVectorizer()
x_train_features= feat_ext.fit_transform(x_train)
x_test_features= feat_ext.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

x_train_features = x_train_features.toarray()
x_test_features = x_test_features.toarray()


#Model Predictions
classifier= GaussianNB()  
classifier.fit(x_train_features, y_train)  
y_pred= classifier.predict(x_test_features)  

#Accuracy Scores
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Precision Score:", precision)
print("Recall score:",recall)

#making predictions
mail=input("Enter the contents of the E-mail:")
mail=[mail]
con_input=feat_ext.transform(mail)
con_input=con_input.toarray()
predictions=classifier.predict(con_input)
if predictions == 1:
    print("This is a genuine E-mail")
else:
    print("This is a spam E-mail")




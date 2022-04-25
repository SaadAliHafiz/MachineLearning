#Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Loading the data
data = pd.read_csv ('heart.csv')

#Print the first 5 rows of the dataset
data.head()
#To get the number of rows and columns in the dataset
data.shape
#Getting information about the data
#data.info()
#Checking the distribution of Target Variable
data['target'].value_counts()

##Splitting the data

X = data.drop (columns='target', axis=1)
Y = data ['target']
#Splitting the Data into Training data & Test Data

X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size=0.2, stratify=Y, random_state=2)
print (X.shape, X_train.shape, X_test.shape)

#Model Linear Regression Model

model = LogisticRegression(solver='lbfgs', max_iter=1000)
# Training the LogisticRegression model with the Training data
model.fit (X_train.values, Y_train)

# Predicting the output with test data
y_pred=model.predict (X_test.values)
print (y_pred)
#Calculating the accuracy of the predicted outcome
print (accuracy_score (Y_test,y_pred))
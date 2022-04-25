#Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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
# print (X.shape, X_train.shape, X_test.shape)

#Model Linear Regression Model

model = LogisticRegression(solver='lbfgs', max_iter=1000)
# Training the LogisticRegression model with the Training data
model.fit (X_train.values, Y_train)

# Predicting the output with test data
y_pred=model.predict (X_test.values)
# print (y_pred)
#Calculating the accuracy of the predicted outcome
#print (accuracy_score (Y_test,y_pred))

##############Predictive System##################################

input_data =[]
# for i in range(13):
#     user_inp = [float(number) if '.' in number else int(number) for number in input("Enter Number: ").replace(',', ' ').split()]
#     input_data.append(user_inp)
age=int(input(" Enter Age: "))
input_data.append(age)
sex=int(input(" Enter sex: "))
input_data.append(sex)
cp=int(input(" Enter cp: "))
input_data.append(cp)
trestbps=int(input(" Enter trestbps: "))
input_data.append(trestbps)
chol=int(input(" Enter chol: "))
input_data.append(chol)
fbs=int(input(" Enter fbs: "))
input_data.append(fbs)
restecg=int(input(" Enter restecg: "))
input_data.append(restecg)
thalach=int(input(" Enter thalach: "))
input_data.append(thalach)
exang=int(input(" Enter exang: "))
input_data.append(exang)
oldpeak=float(input(" Enter oldpeak: "))
input_data.append(oldpeak)
slope=int(input(" Enter slope: "))
input_data.append(slope)
ca=int(input(" Enter ca: "))
input_data.append(ca)
thal=int(input(" Enter thal: "))
input_data.append(thal)

# Change the input data to a numpy array
numpy_data= np.asarray (input_data)
# reshape the numpy array as we are predicting for only on instance
input_reshaped = numpy_data.reshape (1,-1)
prediction = model.predict (input_reshaped)
if (prediction[0]== 0):  
    print ('The Person does not have a Heart Disease')
else:  
    print ('‘The Person has Heart Disease’')
# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.open google colab or jupyter notebook
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset. 
5.Predict the values of array
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. 
7.Apply new unknown values
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: GURUMOORTHI R
RegisterNumber:  212222230042
*/
```
```
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
## Original data(first five columns):

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/2e473c2c-2bbb-4a0d-88a8-2afc3fda4b4e)

## Data after dropping unwanted columns(first five):

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/2257dcb1-360f-436e-aeb6-d03e2fca0fa0)

## Checking the presence of null values:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/8b681a86-680e-47d8-88e1-06e05e711583)

## Checking the presence of duplicated values:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/67dd2a62-c36e-43b5-a8ab-f2d3ce9ee6f3)

## Data after Encoding:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/e5f9deff-a8e9-4496-a9e5-89b566274857)

## X Data:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/b94fb92d-881d-44c0-8737-3bf0cc578394)


## Y Data:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/b87f5c68-9a28-4491-9d4b-e87f5b5a9335)


## Predicted Values:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/2f765029-fe8c-491c-af09-54e9fdb3cfe6)

## Accuracy Score:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/8bdd5339-b012-478a-8c65-d6446df5a41f)

## Confusion Matrix:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/452da6ea-1c2e-48e3-bcbd-d3f648f2a6ff)

## Classification Report:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/f734dac9-db1f-47b9-bf85-3e532ff92ddf)

## Predicting output from Regression Model:

![image](https://github.com/gururamu08/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118707009/614b6d4f-dbf7-4918-91aa-fe0cfacf25c7)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

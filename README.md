# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: gokul prakash
RegisterNumber:  212223240041
*/
```
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
```
```
data = pd.read_csv('Placement_Data.csv')
df = pd.DataFrame(data)
df.head()
```
```
df = df.drop(['sl_no','salary'],axis=1)
df.head()
```
```
df.isnull().sum()
df.duplicated().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ssc_b'] = le.fit_transform(df['ssc_b'])
df['hsc_b'] = le.fit_transform(df['hsc_b'])
df['hsc_s'] = le.fit_transform(df['hsc_s'])
df['degree_t'] = le.fit_transform(df['degree_t'])
df['workex'] = le.fit_transform(df['workex'])
df['specialisation'] = le.fit_transform(df['specialisation'])
df['status'] = le.fit_transform(df['status'])
df.head()
```
```
x = df.drop('status',axis=1)
x.head()
```
```
y = df['status']
y.head()
```
```
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
```
```
reg = LogisticRegression()
reg.fit(x_train,y_train)
```
```
predict =reg.predict(x_test)
predict
```
```
acc=accuracy_score(y_test,predict)
print("Accuracy: {:.3f}".format(acc))
```
```
confusion = confusion_matrix(y_test,predict)
print("Confusion_matrix: ")
print(confusion)
```
```
classification_report = classification_report(y_test,predict)
print("Classification_report: ")
print(classification_report)
```


## Output:
### DATASET
![Screenshot 2025-05-02 202506](https://github.com/user-attachments/assets/0d7ebeda-e9e5-4bd6-ab5e-7f6e7263d5e9)

### X Value
![Screenshot 2025-05-02 202531](https://github.com/user-attachments/assets/2d7eb91b-266b-47a4-b47f-61e458d4b544)

### Y Value
![Screenshot 2025-05-02 202537](https://github.com/user-attachments/assets/de4c81bb-e667-4859-bef7-a4153b1e871e)

### Predict Value
![Screenshot 2025-05-02 202554](https://github.com/user-attachments/assets/b2944b14-ecb4-4460-9b98-6737dd29516a)

### Accuracy Value
![Screenshot 2025-05-02 202559](https://github.com/user-attachments/assets/58663ccc-6fe0-4f4f-a5c3-6d01384a3d4a)

### Confusion Matrix
![Screenshot 2025-05-02 202605](https://github.com/user-attachments/assets/1018ca24-84b0-4823-b324-f2b805aada2f)

### Classification Report
![Screenshot 2025-05-02 202614](https://github.com/user-attachments/assets/826155b7-a162-46f8-bf69-d8f842115d90)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

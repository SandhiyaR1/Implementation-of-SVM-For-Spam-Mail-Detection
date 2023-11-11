# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import chardet

2.Read the dataset

3.Import SVC from sklearn

4.Fit the data in the model and run the algorithm


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SANDHIYA R 
RegisterNumber:  212222230129
*/
```
```
import chardet
file="/content/spam.csv"
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```
## Output:
### Result output
![image](https://github.com/SandhiyaR1/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497571/ae0ec72c-286d-4b66-a002-63a2b9cc269b)

### data.head()
![image](https://github.com/SandhiyaR1/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497571/74dc3961-d293-4f40-a8d9-0c60599a591d)
### data.info()
![image](https://github.com/SandhiyaR1/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497571/04dddd12-9429-4a23-b978-1cfe038931a8)
### data.isnull().sum()
![image](https://github.com/SandhiyaR1/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497571/2f641805-09ca-40ae-9473-f31e5e56bd6b)
### Y_prediction value
![image](https://github.com/SandhiyaR1/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497571/1052240a-9090-449b-af52-115ff42408a4)
### Accuracy value
![image](https://github.com/SandhiyaR1/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113497571/12ef0f32-afc5-42c1-85db-b45ff6e74f62)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

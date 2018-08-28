#QUESTION 1
#Importing the required libraries 
import re
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
#Importing datasets
df=pd.read_csv("https://raw.githubusercontent.com/Shreyas3108/Titanic-EDA-and-Survival-prediction/master/train.csv")
df
##Importing datasets
df1=pd.read_csv("https://raw.githubusercontent.com/Shreyas3108/Titanic-EDA-and-Survival-prediction/master/test.csv")
df1
frames=[df,df1]
result=pd.concat(frames)
result
result.head()
ab=result.dropna(subset=["Survived"])
ab
ab.isnull().sum()
ab.insert(loc=2,column="Has_Cabin",value=ab["Cabin"].notna())
ab
ab["title"]=ab.Name.apply(lambda x:re.search('([A-Z][a-z]+)\.',x).group(1))
ab["title"].unique()
ab
ab["title"]=ab["title"].replace(["Don","Rev","Dr","Major","Lady","Sir","Col","Capt","Countess","Jonkheer"],"special")
ab["title"]
ab["Categorical_age"]=pd.qcut(ab.Age,q=4,labels=False)
ab["Categorical_fare"]=pd.qcut(ab.Fare,q=4,labels=False)
ab.head()
ab=ab.fillna(ab.median())
ab
ab.isnull().sum()
sc=ab.drop(["Cabin","Name","PassengerId","Ticket","Age","Fare","SibSp","Parch"],axis=1)
sc
sc = pd.get_dummies(sc)
sc
X = sc.drop(["Survived"],axis = 1)
Y=sc["Survived"]
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)
reg = LogisticRegression()
reg.fit(X_train,Y_train)
pred = reg.predict(X_test)
result = pd.DataFrame({"Predicted":pred,"Actual":Y_test})
result
reg.score(X_test,Y_test)

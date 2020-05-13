import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data=pd.read_csv('F:\\SONGS\\MachineLearning-master\\DataSets-master\\500_Person_Gender_Height_Weight_Index.csv')
data.head(20)
real_x=data.iloc[:,1:3].values
real_y=data.iloc[:,3]
train_x,test_x,train_y,test_y=train_test_split(real_x,real_y,test_size=0.20)
train_x
from sklearn.ensemble import RandomForestClassifier
reg=RandomForestClassifier(n_estimators=500)
reg.fit(train_x,train_y)
pred=reg.predict(test_x)
pred
from sklearn.metrics import accuracy_score
acc=accuracy_score(pred,test_y)
test_y.head(20)
acc
import pickle
pickle.dump(reg,open("model.pkl","wb"))

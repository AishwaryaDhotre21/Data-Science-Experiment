import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
df=pd.read_csv('Iris.csv',index_col=0)
print(df.head(5))
x1=df.iloc[:,:4]
print("\n\nshape of x1:",x1.shape)
print("\nx1:\n",x1.head(3))
y1=df['Species']
print("Shape of y1:",y1.shape)
print("\ny1:\n",y1.head(3))
x_tr,x_te,y_tr,y_te=train_test_split(x1,y1,test_size=0.20,random_state=1)
print("\nShape\nXtrain:",x_tr.shape,"\nxtest:",x_te.shape,"\nytrain:",y_tr.shape,"\nytest:",y_te.shape)
classifier=GaussianNB()
classifier.fit(x_tr,y_tr)
y_pred=classifier.predict(x_te)
print("\nConfusion matrix:\n",confusion_matrix(y_te,y_pred))

count=0
count1=0

for i in range(len(y_te)):
    if y_pred[i]==y_te.iloc[i]:
        count=count+1
        print("y_pred:",y_pred[i],"=y_test:",y_te.iloc[i],"  count:",count)
    else:
        print("**************************************")
        count1 = count1 + 1
        print("y_pred:", y_pred[i], "=y_test:", y_te.iloc[i], "  count:", count1)
        print("**************************************")


print("\n\nCorrect predicted values:",count)
print("Wrong predicted values:",count1)
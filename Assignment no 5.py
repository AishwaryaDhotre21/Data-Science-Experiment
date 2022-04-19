import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, mean_absolute_error, \
    mean_squared_error

df1=pd.read_csv('Social_Network_Ads.csv')
print(df1.head(5))
df1.dropna(inplace=True)
df1.drop_duplicates(inplace=True)
df2=pd.get_dummies(data=df1['Gender'])
print(df2)
newdf=pd.concat((df1,df2),axis=1)
print(newdf.head(3))
newdf.drop('Gender',axis=1,inplace=True)
print(newdf.head(3))
li=['User ID'  ,'Age'  ,'EstimatedSalary'   , 'Female' , 'Male', 'Purchased']
newdf=newdf[li]
print(newdf.head(3))
newdf.drop('Female',axis=1,inplace=True)
newdf.rename(columns={'Male':'Gender'},inplace=True)
print(newdf.head(3))


scaler=MinMaxScaler(feature_range=(0,1))
newdf['EstimatedSalary']=scaler.fit_transform(newdf[['EstimatedSalary']])
print(newdf.head(3))


features=['Age'  ,'EstimatedSalary'   , 'Gender']
x1=newdf[features]
print("Coloumns :",x1.columns)
y1=newdf['Purchased']
print("Y1 values:",y1)


x_tr,x_te,y_tr,y_te=train_test_split(x1,y1,test_size=0.20,random_state=1)
print("Shape:\n","X_Train:",x_tr.shape,"\nY_train:",y_tr.shape,"\nX_test:",x_te.shape,"\nY_test:",y_te.shape)


model=LogisticRegression()
model.fit(x_tr,y_tr)
y_pred=model.predict(x_te)
print("Shape of y_pred:",y_pred.shape)

print("******************* Measures ******************\n")

print("\nAccuracy score:",accuracy_score(y_te,y_pred)*100)
print("\nPrecision score:",precision_score(y_pred,y_te))
print("Recall score:",recall_score(y_te,y_pred))
print("\nConfusion Matrix:\n",confusion_matrix(y_te,y_pred))
tp,fp,fn,tn=confusion_matrix(y_te,y_pred).ravel()
print(f'Correctly Predicted made Purchase {tp}')
print(f'Falsely Predicted made Purchase {fp}')
print(f'Falsely Predicted made did NOT made Purchase {fn}')
print(f'Correctly Predicted made did NOT made Purchase {tn}')
print("\nModel Training score:",model.score(x_tr,y_tr))
print("\nModel Testing score:",model.score(x_te,y_te))
print(f'{mean_absolute_error(y_te,y_pred)}')
print(f'{mean_squared_error(y_te,y_pred)}')

count=0
count1=0
for i in range(len(y_pred)):
    if y_te.iloc[i]==y_pred[i]:
        count=count+1
        print("y_pred:",y_pred[i]," == y_test:",y_te.iloc[i],"  Count:",count)
    else:
        count1=count1+1
        print("**************************************")
        print("y_pred:", y_pred[i], " == y_test:", y_te.iloc[i], "  Count:", count1)
        print("**************************************")


print("\n\nCorrect predicted values:",count)
print("Wrong predicted values:",count1)
import pandas as pd
import numpy as np
import seaborn as sns
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
obj=LinearRegression()
df=pd.read_csv('boston_housing.csv')
print(df)
print("Coloumns:",df.columns)
df.rename(columns={'medv':'price'},inplace=True)
print("\nAfter modification Coloumns:\n",df.columns)
sns.scatterplot(x='crim',y='price',data=df)
plt.title('relation bet crime rate and house price')
#plt.show()
sns.scatterplot(x='rm',y='price',data=df)
plt.title('relation bet room number and price')
#plt.show()
correlation=df.corr()
print("\n\nCorrelation:\n",correlation)
plt.figure(figsize=(20,20))
#sns.heatmap(correlation,annot=True)
#plt.show()
corr1=df['rm'].corr(df['price'])
print("Correlation betn the room number and price:",corr1)
x1=df.iloc[:,:13]
y1=df['price']
print("********************** Linear Regression ************************\n")
print("Train data input:\n",x1.columns)
print("Train data predicted output:\n",y1)
x_tr,x_te,y_tr,y_te=train_test_split(x1,y1,test_size=0.20,random_state=1)
print("\n\nx-Train data:",x_tr.describe())
print("\n\ny-Train data:",y_tr.describe())
print("\n\nx-Test data:",x_te.describe())
print("\n\ny-test data:",y_te.describe())
print("\n\n************************* Model fitting ************************************\n\n")
obj.fit(x_tr,y_tr)
y_pred=obj.predict(x_te)
print("\n\nY_test data:\n",y_te)
print("\n\ny_pred data:\n",y_pred)
print("\n\nTraining score:",obj.score(x_tr,y_tr)*100)
print("\n\nTesting score:",obj.score(x_te,y_te)*100)
sns.scatterplot(x=y_te,y=y_pred)
plt.title('Comparision bet y_test and y_pred')
#plt.show()
sns.regplot(x=y_te,y=y_pred)
plt.show()

sns.scatterplot(x=y_pred,y=y_te-y_pred)
plt.title("relation bet y_pred and diff of y_test and y_pred")
#plt.show()


new_val=[[0.62976,0.0,8.14 ,0 ,0.538 ,5.949 ,61.8 ,4.7075 ,4 ,307.0 ,21.0 ,396.90 ,8.26]]
y_pred1=obj.predict(new_val)
print("\n\nNew test value:\n",new_val)
print("\n\nPredicted value:",y_pred1)



mse=mean_squared_error(y_te,y_pred)
print("Mean squared error:",mse)


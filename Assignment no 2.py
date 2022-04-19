import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('Modified_campus_placement_data.csv',index_col=0)
print(df)
df2=df.copy()
df.drop(index=0,inplace=True)
print("Coloumns",df.columns)
print("Size:\n",df.size)
print("dimension:\n",df.ndim)
print("Info:\n",df.info)
print("Describe:\n",df.describe())

print("**************************NUll Values**********************")
print("Null Values:\n",df.isnull().sum())
print(df.head(4))
df.fillna(value=df['salary'].mean(),inplace=True)
print("After processing:\n",df.isnull().sum())
print("Null values in df2:\n",df2.isnull().sum())
df2.dropna(inplace=True)
print("After dropping null values in dataframe 2:\n",df2.isnull().sum())

print("**************************Duplicated Values**********************")
print("Duplicated values:\n",df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("After removal of Duplicated values:\n",df.duplicated().sum())

print("**************************Handling the outliers**********************\n\n\n\n\n")
print("*********************** Statistics before modification ****************************")
sns.boxplot(y=df['salary'],data=df)
plt.show()
print('Mean:',df['salary'].mean())
print('median:',df['salary'].median())
print('skew:',df['salary'].skew())
print('std:',df['salary'].std())
print("*********************** Detecting outliers using Interquartile range ****************************")
q1=np.percentile(df['salary'],25,interpolation='midpoint')
print("25th percentile of salary",q1)
q2=np.percentile(df['salary'],50,interpolation='midpoint')
print("50th percentile salary",q2)
q3=np.percentile(df['salary'],75,interpolation='midpoint')
print("75th percentile salary",q3)
q4=np.percentile(df['salary'],100,interpolation='midpoint')
print("100th percentile salary",q4)
IQR=q3-q1
print('Interquartile range:',IQR)
low_lim=q1-1.5*IQR
up_lim=q3+1.5*IQR
print("Lower limit:",low_lim)
print("Upper limit:",up_lim)
print("*********************** Data prior modification ****************************")
outliers=[]
count=0
roll_num=[]
for i in df.index:
    if(df.loc[i,'salary']>up_lim or df.loc[i,'salary']<low_lim):
        outliers.append(df.loc[i,'salary'])
        roll_num.append(df.loc[i,'sl_no'])
        count=count+1

print("Outliers:",outliers)
print("Count of outliers:",count)
print("Roll num of those outliers:",roll_num)
print("Example:\nRoll no",df.loc[4,'sl_no'],"   Salary:",df.loc[4,'salary'])
print("Info:",df.info)
print("*********************** Data after modification ****************************")
count=0
rollnum=[]
sal=[]
for i in df.index:
    if df.loc[i,'salary'] in outliers:
        df.loc[i,'salary']=df['salary'].mean()
        rollnum.append(df.loc[i,'sl_no'])
        sal.append(df.loc[i,'salary'])
        count=count+1

print("Count:",count)
print("Roll num of students whoes salary got replaced:",rollnum)
print("Replaced salary:",sal)
print("Example:\nRoll no",df.loc[4,'sl_no'],"   Salary:",df.loc[4,'salary'])
print("*********************** Statistics after modification ****************************")
q1=np.percentile(df['salary'],25,interpolation='midpoint')
print("25th percentile of salary",q1)
q2=np.percentile(df['salary'],50,interpolation='midpoint')
print("50th percentile salary",q2)
q3=np.percentile(df['salary'],75,interpolation='midpoint')
print("75th percentile salary",q3)
q4=np.percentile(df['salary'],100,interpolation='midpoint')
print("100th percentile salary",q4)
sns.boxplot(y=df['salary'],data=df)
plt.show()
print('Mean:',df['salary'].mean())
print('median:',df['salary'].median())
print('skew:',df['salary'].skew())
print('std:',df['salary'].std())
print("\n\n*********************** Data Normalization ****************************")
df2['salary']=(df2['salary']-df2['salary'].mean())/df2['salary'].std()
print("***********Z-score method of normalization*************\n\n")
print("Normalized salary:",df2['salary'])
df3=df.copy()
print("************Before normalization:***************\n\n",df3['salary'])
df3['salary']=df3['salary'].apply(lambda x: np.log(x) if x!=0 else 0)
print("\n\nlog scalling method of normalization\n\n")
print("Normalized salary:\n\n",df3['salary'])
print("\n\n*********************** Data Transformation ****************************")
df4=pd.read_csv('Modified_campus_placement_data.csv')
df4.drop('Unnamed: 0',axis=1,inplace=True)
print("New dataframe values:\n\n",df4.head(2))
print('Mean:',df4['salary'].mean())
print('median:',df4['salary'].median())
print('skew:',df4['salary'].skew())
print('std:',df4['salary'].std())
df4['salary'].hist()
plt.show()
print("\n\n***********************Quantile calculation****************************")

q0=df4['salary'].quantile(0.01)
print("1st quantile of salary",q0)
q1=df4['salary'].quantile(0.25)
print("25th quantile of salary",q1)
q2=df4['salary'].quantile(0.50)
print("50th quantile salary",q2)
q3=df4['salary'].quantile(0.75)
print("75th quantile salary",q3)
q4=df4['salary'].quantile(1.0)
print("100th quantile salary",q4)

print("\n\n********************Drop rows***********************\n")
print("Before:\n",df4.head(4))
data1=df4.drop(labels=0,inplace=True)
data=df4.drop([1,3,6],inplace=True,axis=0)
print(df4.head(4))
import numpy as np
from sklearn import linear_model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
data=pd.read_csv("insurance.csv")
df=pd.DataFrame(data)
print(df)
df.info()
df.describe()
plt.scatter(data['age'],data['charges'],color='pink')
df.isnull()
df.dropna
regr=linear_model.LinearRegression()
x=np.asanyarray(df['age'])
y=np.asanyarray(df['charges'])
print(x)
print(y)
X=x.reshape(-1,1)
print(X)
out=regr.fit(X,y)
plt.plot(x,regr.predict(X),color='green')

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
x=df[['age','bmi']]
y=np.asanyarray(df['charges'])
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
Y=regr.fit(X_train,y_train)
y_pred=regr.predict(X_test)
#print(y_pred)
b=regr.predict([[30,50]])
print("Charges ".format(b))
#plt.plot(y_test,y_pred,color='g')
plt.scatter(y_test,y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
#Accuracy=r2_score(y_test,y_pred)
#print(" Accuracy of the model is %.2f" %Accuracy)
sns.regplot(x=y_test,y=y_pred,ci=None,color ='red')
plt.show()

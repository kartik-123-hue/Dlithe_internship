import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
dataset=pd.read_csv('Housing.csv')
print(dataset)
X=dataset.iloc[:,1:2].astype(int)
print(X)
y=dataset.iloc[:,2].astype(int)
print(y)
regressor=DecisionTreeRegressor(random_state=0)
clf=regressor.fit(X,y)
tree.plot_tree(clf)
export_graphviz(regressor,out_file='tree1.plot',feature_names=['price'])
y_pred=regressor.predict([[100000]])
print("predicted price: %d"% y_pred)
tree.plot_tree(clf)
plt.title('Profit to Production Cost (Decision Tree Regression)')  
plt.xlabel('Production Cost') 
plt.ylabel('Profit') 
plt.savefig("decision.png") 
plt.show()

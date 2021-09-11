import pandas as pd
data=pd.read_csv("insurance.csv")
data.head()

X=data[['age','bmi']].values
y=data['charges'].values
print(X)
print(y)

from sklearn.linear_model import LinearRegression
regsr=LinearRegression()
regsr.fit(X,y)

#import numpy as np
#prediction=regsr.predict(np.asarray([20,30]).reshape(-1,2))
#print(prediction)

#visualisasing data
from pandas.plotting import scatter_matrix
scatter_matrix(data[['charges','age','bmi', 'children']], alpha=0.3, diagonal='kde')
plt.figure(1)
plt.subplot(2,2,1)
data.groupby(['sex'])['charges'].sum().plot.bar()
plt.subplot(2,2,2)
data.groupby(['smoker'])['charges'].sum().plot.bar()
plt.subplot(2,2,3)
data.groupby(['region'])['charges'].sum().plot.bar()
plt.figure(2)
plt.subplot(2,2,1)
data.groupby(['sex'])['bmi'].sum().plot.bar()
plt.subplot(2,2,2)
data.groupby(['smoker'])['bmi'].sum().plot.bar()
plt.subplot(2,2,3)
data.groupby(['region'])['bmi'].sum().plot.bar()

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 22:04:16 2017

@author: Mac Laptop
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv('train.csv')

df1=df[['Id','LotArea','1stFlrSF','2ndFlrSF','GarageArea','SalePrice']]

df1['TotalSF']=df1['1stFlrSF']+df1['2ndFlrSF']  # Create total sf array
df1=df1.drop(df1.columns[[2,3]],axis=1)  # Drops columns 2 and 3
df1=df1[['Id','LotArea','GarageArea','TotalSF','SalePrice']]  # Set new order of columns

df1=df1.loc[(df1!=0).all(axis=1)]  # Deletes any row with a 0 in any column (to get rid of 0 garage areas)

df2=df1.drop(df1[(df1['LotArea']>50000)].index) # Delete some outliers in lot area

A=df1.as_matrix()  # Converts dataframe to numpy array
B=df2.as_matrix()

plt.scatter(x=df2['GarageArea'],y=df2['SalePrice'])
plt.show()

from sklearn.linear_model import Ridge
from scipy.stats import norm

df2['SalePrice']=np.log1p(df2['SalePrice'])

y=df2['SalePrice']
X=df2[['LotArea','GarageArea','TotalSF']]
clf=Ridge(alpha=0.5)
clf.fit(X,y)

#sns.distplot(df2['SalePrice'] , fit=norm)
"""
Starting test data
"""

dftst=pd.read_csv('test.csv')
dftst1=dftst[['Id','LotArea','1stFlrSF','2ndFlrSF','GarageArea']]
dftst1['TotalSF']=dftst1['1stFlrSF']+dftst1['2ndFlrSF']
dftst1=dftst1.drop(dftst1.columns[[2,3]],axis=1)

X1=dftst1[['LotArea','GarageArea','TotalSF']]
X11=X1.as_matrix()
X11[np.isnan(X11)]=0
D=clf.predict(X11)
D=np.expm1(D)  # Back transform results that arein log1p form

plt.scatter(range(1,1460),D)
plt.show()

# Create dataframe with D values and id's from test data
d={'SalePrice' : D}
dfd=pd.DataFrame(d)
dfd['Id']=dftst[['Id']]
dfd=dfd[['Id','SalePrice']]
dfd.to_csv('RidgeRegPyLog.csv',index=False)

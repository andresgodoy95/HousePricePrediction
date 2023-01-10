#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the Boston House Pricing Dataset

# In[2]:


from sklearn.datasets import load_boston


# In[3]:


boston = load_boston()


# In[4]:


boston.keys()


# In[5]:


##Lets check the description of the dataset
print(boston.DESCR)


# In[6]:


print(boston.data)


# In[7]:


dataset = pd.DataFrame(boston.data,columns = boston.feature_names)
dataset1 = pd.DataFrame(boston.data,columns = boston.feature_names)


# In[8]:


dataset.head()


# In[9]:


dataset['Price'] = boston.target


# In[10]:


dataset.head()


# In[11]:


dataset.info()


# In[12]:


dataset.describe()


# # Check the missing Values

# In[13]:


dataset.isnull().sum()


# # Exploratory Data Analysis

# Correlation

# In[14]:


dataset.corr()


# In[15]:


plt.scatter(dataset['CRIM'], dataset['Price'])
plt.xlabel('Crime Rate')
plt.ylabel('Price')


# In[16]:


plt.scatter(dataset['RM'], dataset['Price'])
plt.xlabel('RM')
plt.ylabel('Price')


# In[17]:


import seaborn as sns
sns.regplot(x='RM', y='Price', data = dataset)


# In[18]:


sns.regplot(x='LSTAT', y='Price', data = dataset)


# In[19]:


sns.regplot(x='CHAS', y='Price', data = dataset)


# In[20]:


sns.regplot(x='PTRATIO', y='Price', data = dataset)


# In[21]:


## Independent and dependent features

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]


# In[22]:


x.head()


# In[23]:


y


# In[24]:


## Train test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x,y,test_size=0.3,random_state=42)


# In[25]:


x_train


# In[26]:


x_test


# # Standarize the dataset

# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[43]:


X_train = scaler.fit_transform(x_train)


# In[45]:


X_test = scaler.fit_transform(x_test)


# In[30]:


x_train


# # Model Training

# In[47]:


from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm


# In[48]:


# Modeling
# 1. OLS
regression = LinearRegression() 
regression.fit(x_train,y_train)
reg_pred = regression.predict(X_test)

# 2. Ridge
ridge = Ridge(alpha = 0.5)
ridge.fit(X_train, y_train)
ridge_yhat = ridge.predict(X_test)

# 3. Lasso
lasso = Lasso(alpha = 0.01)
lasso.fit(X_train,y_train)
lasso_yhat = lasso.predict(X_test)

# 4. Bayesian
bayesian = BayesianRidge()
bayesian.fit(X_train, y_train)
bayesian_yhat = bayesian.predict(X_test)

# 5. ElasticNet
en = ElasticNet(alpha = 0.01)
en.fit(X_train, y_train)
en_yhat = en.predict(X_test)


# In[52]:


## plot a scatter plot for the prediction
plt.scatter(y_test, reg_pred)
plt.show()
plt.scatter(y_test, ridge_yhat)
plt.show()
plt.scatter(y_test, lasso_yhat)
plt.show()
plt.scatter(y_test, bayesian_yhat)
plt.show()
plt.scatter(y_test, en_yhat)
plt.show()


# In[54]:


## residuals or error
residuals_ols = y_test - reg_pred
residuals_ridge = y_test - ridge_yhat
residuals_lasso = y_test - lasso_yhat
residuals_bayesian = y_test - bayesian_yhat
residuals_en = y_test - en_yhat


# In[55]:


## plot the residuals
sns.displot(residuals_ols, kind = 'kde')
sns.displot(residuals_ridge, kind = 'kde')
sns.displot(residuals_lasso, kind = 'kde')
sns.displot(residuals_bayesian, kind = 'kde')
sns.displot(residuals_en, kind = 'kde')
plt.show()


# In[42]:


## scatter plot with respect to prediction and residuals (uniform dist)
plt.scatter(reg_pred, residuals)


# In[62]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, reg_pred))
print(mean_squared_error(y_test,reg_pred))
print(np.sqrt(mean_squared_error(y_test,reg_pred)))


# # R square and adj R square

# In[63]:


from sklearn.metrics import r2_score as r2 # evaluation metric


# In[66]:


# 2. R-squared

print('R-SQUARED:')
print('-------------------------------------------------------------------------------')
print('R-Squared of OLS model is {}'.format(r2(y_test, reg_pred)))
print('-------------------------------------------------------------------------------')
print('R-Squared of Ridge model is {}'.format(r2(y_test, ridge_yhat)))
print('-------------------------------------------------------------------------------')
print('R-Squared of Lasso model is {}'.format(r2(y_test, lasso_yhat)))
print('-------------------------------------------------------------------------------')
print('R-Squared of Bayesian model is {}'.format(r2(y_test, bayesian_yhat)))
print('-------------------------------------------------------------------------------')
print('R-Squared of ElasticNet is {}'.format(r2(y_test, en_yhat)))
print('-------------------------------------------------------------------------------')


# In[78]:


a = [reg_pred,ridge_yhat,lasso_yhat,bayesian_yhat,en_yhat]
for i in range(len(a)):
    adj_ols = 1 - (1-r2(y_test,a[i]))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
    print('Adj R-square of ',i,':',adj_ols )

#adj_ols = 1 - (1-r2(y_test,reg_pred))*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
#print('Adj R-square is:',adj_ols)


# # New data prediction

# In[156]:


boston.data[2]


# In[105]:


scaler.transform(boston.data[2].reshape(1,-1))


# In[106]:


regression.predict(scaler.transform(boston.data[2].reshape(1,-1)))


# In[80]:


df2 = pd.DataFrame(y_test)
df2['OLS'] = reg_pred
df2['Ridge'] = ridge_yhat
df2['Lasso'] = lasso_yhat
df2['Bayesian'] = bayesian_yhat
df2['Elastic Net'] = en_yhat
df2


# In[83]:


# set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above) 
sns.set(style="darkgrid")

# plotting both distibutions on the same figure
fig = sns.kdeplot(df2['Price'], shade=True, color="r")
fig = sns.kdeplot(df2['OLS'], shade=True, color="b")
fig = sns.kdeplot(df2['Ridge'], shade=True, color="g")
fig = sns.kdeplot(df2['Lasso'], shade=True, color="yellow")
fig = sns.kdeplot(df2['Bayesian'], shade=True, color="purple")
fig = sns.kdeplot(df2['Elastic Net'], shade=True, color="grey")
plt.title('Model Test')
plt.show()


# The best model for the prediction is OLS Regression, as it scores the max R2 and adj R2 of %66.61

# In[ ]:





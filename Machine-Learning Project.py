#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# In[3]:


raw_data = pd.read_csv('C:\\Users\\Asus\\Desktop\\MachineLearning-Coursera\\1.03. Dummies.csv')
raw_data


# In[4]:


data = raw_data.copy()


# In[5]:


data['Attendance'] = data['Attendance'].map({'Yes' : 1, 'No': 0})


# In[6]:


data


# In[7]:


data.describe()


# In[8]:


##Regression


# In[11]:


y = data['GPA']
x1 = data[['SAT','Attendance']]


# In[12]:


x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()


# In[13]:


plt.scatter(data['SAT'],y)
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


# In[14]:


plt.scatter(data['SAT'],y,c=data['Attendance'],cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


# In[15]:


plt.scatter(data['SAT'],data['GPA'], c=data['Attendance'],cmap='RdYlGn_r')
yhat_no = 0.6439 + 0.0014*data['SAT']
yhat_yes = 0.8665 + 0.0014*data['SAT']
yhat = 0.0017*data['SAT'] + 0.275
fig = plt.plot(data['SAT'],yhat_no, lw=2, c='#006837', label ='regression line1')
fig = plt.plot(data['SAT'],yhat_yes, lw=2, c='#a50026', label ='regression line2')
fig = plt.plot(data['SAT'],yhat, lw=3, c='#4C72B0', label ='regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()


# In[16]:


x


# In[18]:


new_data = pd.DataFrame({'const':1, 'SAT':[1700, 1670], 'Attendance': [0,1]})
new_data = new_data[['const', 'SAT', 'Attendance']]
new_data


# In[19]:


new_data.rename(index= {0: 'Bob', 1: 'Alice'})


# In[20]:


predictions = results.predict(new_data)
predictions


# In[21]:


predictionsdf = pd.DataFrame({'Predictions': predictions})
joined = new_data.join(predictionsdf)
joined.rename(index= {0: 'Bob', 1: 'Alice'})


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data pre-processing

dataset = pd.read_csv(r'C:\Users\user\Downloads\Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(np.nan, 'mean')
# imputer.fit(x[:,:])
# x[:,:] = imputer.transform(x[:,:])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state = 1)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train[:, :] = sc.fit_transform(x_train[:, :])
# x_test[:, :] = sc.transform(x_test[:, :])

#model training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# predict
y_pred = regressor.predict(x_test)

#visualize






# In[2]:


plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary Vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


# In[3]:


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary Vs Experience (test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





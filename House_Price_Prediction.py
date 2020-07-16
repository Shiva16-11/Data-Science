#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import os


# In[2]:


os.chdir("D:\R_training\House Price")
os.getcwd()


# In[3]:


dataset = pd.read_csv("train.csv")
print(dataset.shape)


# In[4]:


pd.pandas.set_option('display.max_columns', None)


# In[5]:


dataset.head()


# In[6]:


Columns_with_null = [x for x in dataset.columns if dataset[x].isnull().sum() > 1 ]

for column in Columns_with_null:
    print(column, np.round(dataset[column].isnull().mean(), 4))


# In[7]:


for column in Columns_with_null:
    data = dataset.copy()
    data[column] = np.where(data[column].isnull(), 1, 0)
    data.groupby(column)['SalePrice'].mean().plot.bar()
    plt.title("Column")
    plt.show()


# In[8]:


''' the missing values have some kind of relationship with the dependent variable'''


# In[9]:


numerical_variables = [x for x in dataset.columns if dataset[x].dtype != 'O']
print(len(numerical_variables))
year_variables = [x for x in numerical_variables if 'Year' in x or 'Yr' in x]
print(year_variables)


# In[10]:


for x in year_variables:
    data = dataset.copy()
    data.groupby(x)['SalePrice'].mean().plot()
    plt.xlabel(x)
    plt.ylabel('SalePrice')
    plt.title("House price Vs Year")
    plt.show()


# In[11]:


discreet_variables = [x for x in numerical_variables if len(dataset[x].unique()) < 35 and x not in year_variables]
print(len(discreet_variables))


# In[12]:


continuous_variables = [x for x in numerical_variables if x not in discreet_variables+year_variables and x != 'ID']
print(len(continuous_variables))


# In[13]:


# check if continuous variables are normally distributed
for x in continuous_variables:
    data = dataset.copy()
    data.groupby(x)['SalePrice'].mean().hist()
    plt.xlabel(x)
    plt.ylabel('SalePrice')
    plt.show()


# In[14]:


#use lograthmin transformation to make distribution normal
for x in continuous_variables:
    data = dataset.copy()
    if 0 in data[x].unique():
        pass
    else:
        data[x] = np.log(data[x])
        data['SalePrice'] = np.log(data['SalePrice'])
        plt.scatter(data[x], data['SalePrice'])
        plt.xlabel(x)
        plt.ylabel('SalePrice')
        plt.title(x)
        plt.show()


# In[15]:


# find outliers

for x in continuous_variables:
    data = dataset.copy()
    if 0 in data[x].unique():
        pass
    else:
        
        data[x] = np.log(data[x])
        data.boxplot(column = x)
        
        plt.ylabel(x)
        plt.title(x)
        plt.show()


# In[16]:


## categorical variables

categorical_variables = [x for x in dataset.columns if dataset[x].dtypes == 'O']
print(len(categorical_variables))


# In[17]:


for x in categorical_variables:
    print(x, len(dataset[x].unique()))


# In[18]:


##  check if there is any relationship between SalePrice and categorical features

for x in categorical_variables:
    data = dataset.copy()
    data.groupby(x)['SalePrice'].mean().plot.bar()
    plt.xlabel(x)
    plt.ylabel('SalePrice')
    plt.title(x)
    plt.show()


# In[19]:


## Feature Engineering
#handle missing categorical value
cat_variable_null = [x for x in dataset.columns if dataset[x].isnull().sum() > 1 and dataset[x].dtypes == 'O']
for x in cat_variable_null:
    print(x, np.round(dataset[x].isnull().mean(), 4))


# In[20]:


# replace null values
def replace_null_categorical(dataset, cat_variable_null):
    data = dataset.copy()
    data[cat_variable_null] = data[cat_variable_null].fillna('Missing')
    return data

dataset = replace_null_categorical(dataset, cat_variable_null)
dataset[cat_variable_null].isnull().sum()
    


# In[21]:


# numerical variable that contain missing values
num_variable_null = [x for x in dataset.columns if dataset[x].isnull().sum() > 1 and dataset[x].dtypes != 'O']
for x in num_variable_null:
    print(x, np.round(dataset[x].isnull().mean(), 4))


# In[22]:


# replaace numerical null values
for x in num_variable_null:
    mean = dataset[x].mean()
    dataset[x+'_'+'null'] = np.where(dataset[x].isnull, 1, 0)
    dataset[x].fillna(mean, inplace = True)
dataset[num_variable_null].isnull().sum()


# In[23]:


for x in ['YearRemodelled', 'YearBuilt', 'GarageYearBuilt']:
    dataset[x] = dataset['YearSold'] - dataset[x]


# In[24]:


num_variables = [x for x in dataset.columns if dataset[x].dtypes != 'O']
for x in num_variables:
    if 0 in dataset[x].unique():
        pass
    else:
        dataset[x] = np.log(dataset[x])
dataset.head()


# In[25]:


# rare categorical features
cat_variables = [x for x in dataset.columns if dataset[x].dtypes == 'O']
for x in cat_variables:
    t = dataset.groupby(x)['SalePrice'].count()/len(dataset['SalePrice'])
    temp = t[t > 0.05].index
    dataset[x] = np.where(dataset[x].isin(temp), dataset[x], "Rare_Variable")


# In[26]:


# feature scaaling
variable_to_scale = [x for x in dataset.columns if x not in ['ID', "SalePrice"] and dataset[x].dtypes != 'O']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
data = sc.fit_transform(dataset[variable_to_scale])


# In[27]:


df = pd.concat([dataset[['ID', 'SalePrice']].reset_index(drop = True), pd.DataFrame(data, columns = variable_to_scale)], axis = 1)
df[df==np.inf]=np.nan
df.fillna(df.mean(), inplace=True)


# In[28]:


#feature selection

X = df.drop(['ID', 'SalePrice'], axis = 1)
Y = df['SalePrice']

from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

 


# In[29]:


feature_selection_model = SelectFromModel(Lasso(alpha = 0.005, random_state = 0))
feature_selection_model.fit(X_train, Y_train)
feature_selection_model.fit(X_test, Y_test)


# In[30]:


selected_features = X_train.columns[(feature_selection_model.get_support())]
selected_features_test = X_test.columns[(feature_selection_model.get_support())]
X_train = X_train[selected_features]
X_test = X_test[selected_features_test]


# In[31]:


#model training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# predict
y_pred = regressor.predict(X_test)


# In[32]:


#calculate RMSE
y_pred = pd.DataFrame(y_pred, columns = ['Predicted Result'])
Y_test = pd.DataFrame(Y)
final = pd.concat([Y_test, y_pred], axis = 1)
final['Error'] = ((final['SalePrice'] - final['Predicted Result']) ** 2)


# In[33]:


RMSE = final['Error'].mean() ** 0.5
RMSE


# In[34]:


final['MAPE'] = (final['SalePrice'] - final['Predicted Result'])/final['SalePrice']


# In[35]:


final['MAPE'].mean()


# In[36]:


final['APE'] = (final['SalePrice'] - final['Predicted Result'])


# In[37]:


final['APE'].mean()


# In[38]:


#model training
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)
# predict
y_pred = regressor.predict(X_train)


# In[39]:


#calculate RMSE
y_pred = pd.DataFrame(y_pred, columns = ['Predicted Result'])
Y_test = pd.DataFrame(Y)
final = pd.concat([Y_train, y_pred], axis = 1)
final['Error'] = ((final['SalePrice'] - final['Predicted Result']) ** 2)


# In[ ]:


RMSE = final['Error'].mean() ** 0.5
RMSE


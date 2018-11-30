
# coding: utf-8

# # <h1><center>Car Price Prediction</center></h1>

# ### 1. Load the library and data

# In[4]:


import pandas as pd
import numpy as np
import matplotlib
get_ipython().magic('matplotlib inline')
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


# In[63]:


train = pd.read_csv("train.csv", encoding='latin-1')
test = pd.read_csv("test.csv",encoding='latin-1')
sample_soln = pd.read_csv("solution.csv",encoding='latin-1')


# In[64]:


train.head()


# In[65]:


test.head()


# In[66]:


train.info()


# In[67]:


test.info()


# In[68]:


train["source"] = 'train'
test["source"] = 'test'


# In[69]:


dataset = pd.concat([train,test],axis=0)


# **See How Many Missing Values We Have**

# In[70]:


Missing_Data=dataset.isnull().sum()
Missing_Data


# In[71]:


dataset.info()


# In[72]:


col_category = dataset.select_dtypes(exclude=[np.number]).columns
col_category


# In[73]:


for col in col_category:
    print ([col]," : ",dataset[col].unique(),'\n')


# **Drop Unuseful COlumns**

# In[74]:


drop_cols=["x2","x3","x10", "x16", "x19","x14","x4","x5"]


# In[75]:


dataset=dataset.drop(drop_cols,axis=1)


# **Assigning Dummies to NaN values**

# In[76]:


dataset['x13'].fillna(value='blank', inplace=True)
dataset['x15'].fillna(value='blank', inplace=True)
dataset['x6'].fillna(value='blank', inplace=True)
dataset['x8'].fillna(value='blank', inplace=True)


# In[77]:


dataset.head()


# In[78]:


col_category = dataset.select_dtypes(exclude=[np.number]).columns


# In[79]:


for col in col_category:
    print ([col]," : ",dataset[col].unique(),'\n')


# In[80]:


col_Integer = dataset.columns.difference(col_category)


# In[81]:


dataset_OHE= pd.get_dummies(dataset[col_category])


# In[82]:


final_df = pd.concat([dataset_OHE,dataset[col_Integer]],axis=1)
final_df


# In[95]:


train_modified = final_df.loc[final_df.source_train==1.0,:]


# In[84]:


test_modified = final_df.loc[final_df.source_train==0.0,:]


# In[92]:


test_modified=test_modified.drop(['source_test', 'source_train','y'],axis=1)


# In[93]:


train_modified=train_modified.drop(['source_test', 'source_train'],axis=1)


# In[94]:


train_modified.head()


# In[96]:


train_modified.to_csv("train_ready_for_model.csv",index=False)


# In[97]:


test_modified.to_csv("test_ready_for_model.csv",index=False)


# In[98]:


X = train_modified.drop(['x1','y'],axis = 1)
y = train_modified['y']

print(X.shape)
print(y.shape)


# In[100]:


from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.33,random_state = 0)


# In[105]:


from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


# In[106]:


models = []
models.append(('LR', LinearRegression()))
models.append(('LSR', Lasso()))
models.append(('RD', Ridge()))
#models.append(('SVR', SVR()))
models.append(('KNR', KNeighborsRegressor()))
models.append(('DTR', DecisionTreeRegressor()))
#models.append(('ETR', ExtraTreesRegressor()))
models.append(('RFR', RandomForestRegressor()))
models.append(('ADA', AdaBoostRegressor()))
models.append(('GRA', GradientBoostingRegressor()))
models.append(('XG', XGBRegressor()))
#models.append(('LGBM', LGBMRegressor()))

# evaluate each model in turn
for name, model in models:
    model.fit(X_train,y_train)
    y_predict_1 = model.predict(X_valid)
    y_predict_train_1 = model.predict(X_train)
    #MSE_CV = np.mean(-cross_val_score(model,X_train,y_train,cv = 10,scoring = 'neg_mean_squared_error'))
    #RMSE_CV = np.sqrt(MSE_CV)
    #RMSE = np.sqrt(metrics.mean_squared_error(y_valid,y_predict_1))
    MAPE_test = np.mean(np.abs((y_valid - y_predict_1) / y_valid)) * 100
    MAPE_train = np.mean(np.abs((y_train - y_predict_train_1) / y_train)) * 100
    print(name,':','\n','Mean Absolute Percentage Error for Train data:',MAPE_train,'\n','Mean Absolute Percentage Error for Valid data:',MAPE_test)


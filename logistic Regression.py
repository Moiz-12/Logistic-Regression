#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


data=pd.read_csv('iris.csv')
df=data.copy()


# In[3]:


df.head()


# In[5]:


df.tail()


# In[6]:


df['species'].unique()


# In[7]:


df.replace('setosa',1,inplace=True)
df.replace('versicolor',2,inplace=True)
df.replace('virginica',3,inplace=True)


# In[8]:


df['species'].unique()


# In[9]:


df


# In[10]:


X=df.drop('species',axis=1)


# In[11]:


y=df['species']


# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=424)


# In[13]:


X_train.shape


# In[14]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import cross_val_score


# In[15]:


clf=LogisticRegression()


# In[16]:


X_train


# In[17]:


clf.fit(X_train, y_train)


# In[18]:


pro=pd.DataFrame(np.array([5.0,3.9,1.8,0.23]).reshape(-1,4),columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])


# In[19]:


clf.predict(pro)


# In[21]:


predicted= clf.predict(X_test)
print(predicted)


# In[22]:


Zp=clf.decision_function(X_test)


# In[23]:


accuracy_score(y_test, predicted)*100


# In[ ]:





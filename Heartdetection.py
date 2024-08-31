#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[2]:


data = pd.read_csv('heart.csv')


# In[3]:


data


# ### data columns information
# 
# 
# Age: Age
# 
# Sex: Sex (1 = male; 0 = female)
# 
# ChestPain: Chest pain (typical, asymptotic, nonanginal, nontypical)
# 
# RestBP: Resting blood pressure
# 
# Chol: Serum cholestoral in mg/dl
# 
# Fbs: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
# 
# RestECG: Resting electrocardiographic results
# 
# MaxHR: Maximum heart rate achieved
# 
# ExAng: Exercise induced angina (1 = yes; 0 = no)
# 
# Oldpeak: ST depression induced by exercise relative to rest
# 
# Slope: Slope of the peak exercise ST segment
# 
# Ca: Number of major vessels colored by flourosopy (0 - 3)
# 
# Thal: (3 = normal; 6 = fixed defect; 7 = reversable defect)
# 
# target: AHD - Diagnosis of heart disease (1 = yes; 0 = no)
# 

# In[4]:


data.info()


# ## Univariat Analysis

# In[5]:


sns.barplot(data['sex'],data['target'] )


# In[6]:


dat2 = data[['target','restecg']].groupby(['restecg','target']).size().reset_index(name='count')


# In[7]:


dat2


# In[8]:


pivoted = dat2.pivot(index='restecg', columns='target', values='count')

# Create the multi bar barplot
pivoted.plot(kind='bar', stacked=True)

# Add labels and title
plt.xlabel('restecg')
plt.ylabel('count')
plt.title('Count of target by restecg level')
plt.legend(title='target')
plt.show()


# In[9]:


fig, ax = plt.subplots(2, 2, figsize=(12, 10))

sns.histplot(data[data['target']==0].age, color='lightblue', ax=ax[0, 0])
ax[0, 0].set_title('Age of patients without heart disease')

sns.histplot(data[data['target']==1].age, color='red', ax=ax[0, 1])
ax[0, 1].set_title('Age of patients with heart disease')

sns.histplot(data[data['target']==0].thalach, color='lightblue', ax=ax[1, 0])
ax[1, 0].set_title('Max heart rate of patients without heart disease')

sns.histplot(data[data['target']==1].thalach, color='red', ax=ax[1, 1])
ax[1, 1].set_title('Max heart rate of patients with heart disease')

plt.show()


# ## Multivariat analysis pair plot

# In[10]:


sns.pairplot(data)


# ## Co relation multivariat analysis

# In[11]:


plt.figure(figsize=(15,10))
sns.heatmap(data.corr(),annot = True)


# In[ ]:





# ## splitting the data

# In[12]:


x = data.drop('target',axis = 1)
y = data['target']


# In[13]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,shuffle = True)


# In[14]:


print('shape of training feature is', xtrain.shape)
print('shape of test feature is', xtest.shape)


# ## Model 1 Random Forest

# In[15]:


rf=RandomForestClassifier()
rf.fit(xtrain,ytrain)
RandomForestClassifier()
prediction=rf.predict(xtest)
prediction
rf_accuracy=accuracy_score(ytest,prediction)*100
rf_accuracy


# ## Model 2 KNN

# In[16]:



acc = []
# Will take some time

for i in range(1,40):
    neigh = KNeighborsClassifier(n_neighbors = i).fit(xtrain,ytrain)
    yhat = neigh.predict(xtest)
    acc.append(metrics.accuracy_score(ytest, yhat))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,40),acc,color = 'blue', 
         marker='o',markerfacecolor='black', markersize=10)
plt.title('accuracy vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))

KNN_accuracy = max(acc)*100


# ## Model Comparision

# In[17]:


algorithms=['Random Forest','KNN']
scores=[rf_accuracy,KNN_accuracy]
sns.set(rc={'figure.figsize':(10,10)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")
sns.barplot(algorithms,scores , palette='Set2')


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go
import time
init_notebook_mode(connected=True)


# In[4]:


def sigmoid(X,weight):
    z = np.dot(X,weight)
    return 1/(1+np.exp(-z))


# In[5]:


def loss(h,y):
    return (-y*np.log(h)-(1-y)*np.log(1-h)).mean()


# In[6]:


def gradient_descent(X,h,y):
    return np.dot(X.T,(h-y))/y.shape[0]
def update_weight_loss(weight,learning_rate,gradient):
    return weight -learning_rate*gradient


# In[7]:


data = pd.read_csv(r'C:\Users\Nalin\Desktop\heart.csv')
print("Dataset size")
print("Rows:{} Columns:{}".format(data.shape[0], data.shape[1]))


# In[8]:


print("Columns and Data types")
pd.DataFrame(data.dtypes).rename(columns={0:'dtype'})


# In[9]:


df = data.copy()


# In[11]:


X = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
X2 = df[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']].copy()
y=df['target'].copy() 


# In[16]:


start_time = time.time()
num_iter = 1000
intercept = np.ones((X.shape[0], 1))
X = np.concatenate((intercept,X), axis=1)
theta = np.zeros(X.shape[1])
for i in range(num_iter):
    h=sigmoid(X,theta)
    gradient = gradient_descent(X,h,y)
    theta = update_weight_loss(theta, 0.1, gradient)
print("training time (log reg using gradient descent ):"+str(time.time()-start_time)+"seconds")
print("Learning rate: {} \n Iteration: {}".format(0.1, num_iter))


# In[17]:


result = sigmoid(X,theta)


# In[19]:


f = pd.DataFrame(np.around(result, decimals=6)).join(y)
f['pred'] = f[0].apply(lambda x:0 if x<0.5 else 1)
print("Accuracy (Loss minimisation):")
f.loc[f['pred']==f['target']].shape[0]/f.shape[0]*100


# In[20]:


df.info()


# In[21]:


df.describe()


# In[24]:


import seaborn as sns
plt.figure(figsize=(10,8), dpi=80)
sns.heatmap(df.corr(), cmap='RdYlGn', center=0)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()


# In[26]:


sns.distplot(df['target'], rug=True)
plt.show()


# In[30]:


import plotly
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
col="target"
colors=['pink', 'silver']
grouped= df[col].value_counts().reset_index()
grouped = grouped.rename(columns={col: "count", "index": col})
trace = go.Pie(labels=grouped[col], values=grouped['count'],pull=[0.05,0], marker=dict(colors=colors, line=dict(color='#000000', width=2)))
layout = {'title': 'Target(0=No, 1=Yes)'}
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[31]:


sns.distplot(df['sex'], rug=True)
plt.show()


# In[35]:


col = "sex"
colors = ['red', 'black']
grouped = df[col].value_counts().reset_index()
grouped = grouped.rename(columns={col: "count","index": col })
trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05,0], marker=dict(colors=colors, line=dict(color='#000000', width=2)))
layout = {'title':'Male(1), Female(0)'}
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[36]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,5))
sns.countplot(x=df.target, hue=df.sex)
plt.legend(labels=['Female', 'Male'])


# In[37]:


dy = pd.DataFrame(df.groupby('sex')['target'].mean().reset_index().values, columns=["gender", "target1"])
dy.head()


# In[39]:


sns.barplot(dy.gender,dy.target1)
plt.ylabel('rate of heart attack')
plt.title('0=Female, 1=Male')


# In[40]:


sns.distplot(df['cp'], rug=True)
plt.show()


# In[43]:


content=df['cp'].value_counts().to_frame().reset_index().rename(columns={'index':'c1', 'C1':'count'})
fig = go.Figure([go.Pie(labels=content['c1'], values=content['cp'], hole=0.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=15)
fig.update_layout(title="Chest Pain Types", title_x=0.5)
fig.show()


# In[44]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(8,5))
sns.countplot(x=df.target, hue=df.cp)
plt.legend(labels=['0: typical angina', '1:atypical angina', '2: non-aginal pain', '3: asymptomatic'])


# In[45]:


dy=pd.DataFrame(df.groupby('cp')['target'].mean().reset_index().values, columns=['chest_pain', 'target2'])
dy.head()


# In[46]:


sns.barplot(dy.chest_pain, dy.target2)
plt.ylabel('rate of heart attack')


# In[47]:


sns.distplot(df.thalach, rug=True)
plt.show()


# In[51]:


col = 'thalach'
d1 = df[df['target']==0]
d2 = df[df['target']==1]
v1 =d1[col].value_counts().reset_index()
v1 = v1.rename(columns= {col : "count","index": col})
v1['percent']=v1['count'].apply(lambda x: 100*x/sum(v1['count']))
v1 = v1.sort_values(col)
v2 = d2[col].value_counts().reset_index()
v2 = v2.rename(columns={col: "count", "index": col})
v2['percent']= v2['count'].apply(lambda x: 100*x/sum(v2['count']))
v2 = v2.sort_values(col)
trace1 = go.Scatter(x=v1[col], y=v1["count"], name=0, mode='lines+markers')
trace2 = go.Scatter(x=v2[col], y=v2["count"], name=1, mode='lines+markers')
data = [trace1, trace2]
layout = {'title': "target over the person's max heart rate achieved ", 'xaxis':{'title':"Thalach"}}
fig = go.Figure(data, layout=layout)
iplot(fig)


# In[ ]:





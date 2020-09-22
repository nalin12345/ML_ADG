#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
points = pd.read_csv(r"C:\Users\Nalin\Desktop\insurance.csv")
#points.drop(['age'], axis=1)
Points = points[['bmi', 'charges']]
print(Points)


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
x = points['bmi']
y = points['charges']
plt.plot(x,y,'o')
plt.show()


# In[8]:


def cost(X,Y,theta):
    J = np.dot((np.dot(X,theta)-Y).T,(np.dot(X,theta)-Y))/(2*len(Y))
    return J
alpha = 0.1
theta = np.array([[0,0]]).T
X = np.c_[np.ones(1338), x]
Y = np.c_[y]
X_1 = np.c_[x].T
num_iters = 1000
cost_history = []
theta_history = []
for i in range(num_iters):
    a = np.sum(theta[0]-alpha*(1/len(Y))*np.sum((np.dot(X,theta)-Y)))
    b = np.sum(theta[1]-alpha*(1/len(Y))*np.sum(np.dot(X_1,(np.dot(X,theta)-Y)))) 
    theta = np.array([[a],[b]])
    cost_history.append(cost(X,Y,theta))
    theta_history.append(theta)
    if i in (1,3,7,10,14,num_iters):
        plt.plot(x,a+x*b)
        plt.suptitle('Linear regression by gradient descent')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    elif i in range(20,num_iters,10):
        plt.plot(x,a+x*b)
    print(theta)


# In[ ]:





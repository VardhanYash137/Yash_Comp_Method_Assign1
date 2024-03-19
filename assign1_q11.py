#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np

# first system
A1= np.array([[3,-1,1],[3,6,2],[3,3,7]])
b1= np.array([1,0,4])
print("a) x: ",np.linalg.solve(A1,b1))

#second system
A2= np.array([[10,-1,0],[-1,10,-2],[0,-2,10]])
b2= np.array([9,7,6])
print("b) x: ",np.linalg.solve(A2,b2))

#third system
A3= np.array([[10,5,0,0],[5,10,-4,0],[0,-4,8,-1],[0,0,-1,5]])
b3= np.array([6,25,-11,-11])
print("c) x: ",np.linalg.solve(A3,b3))

#fourth system
A4= np.array([[4,1,1,0,1],[-1,-3,1,1,0],[2,1,5,-1,-1],[-1,-1,-1,4,0],[0,2,-1,1,4]])
b4= np.array([6,6,6,6,6])
print("c) x: ",np.linalg.solve(A4,b4))


# In[ ]:





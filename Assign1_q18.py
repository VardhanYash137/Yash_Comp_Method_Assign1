#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np

def power_method(A,x0=None,max_iter=100,tol=10**-2):
    
    k=0
    if x0==None:
        x0=np.ones(len(A))
    eig_v=np.ones(max_iter)  #storing all the eigenvalues of each iteration
    
    while k<max_iter:
        
        #Calculation of eigenvalues using power method algorithm
        Ax=np.dot(A,x0)
        A2x=np.dot(A,Ax)
        eig_val=np.dot(A2x,Ax)/np.dot(Ax,Ax)
        eig_v[k]=eig_val
        
        #Calculation of relative error then compared with tolerance
        if k>0:
            if np.abs(eig_v[k]-eig_v[k-1])/np.abs(eig_v[k])<tol:
                return eig_v[k],(1/np.linalg.norm(Ax))*Ax,k+1 #second element returned is the normalised eigenvector
        
        #Update x0 and counter
        x0=np.dot(A,x0).copy()
        k=k+1
        
    return eig_vec[k],(1/np.linalg.norm(Ax))*Ax,k+1,"tolerance not reached"

A=np.array([[2,-1,0],[-1,2,-1],[0,-1,2]])
print("Eigenvalues using Power method: ",power_method(A)[0])
print("Eigenvector using Power method: ",power_method(A)[1])
print("Iterations required for 1% precision: ",power_method(A)[2])

print("Inbuilt function: ",np.linalg.eigh(A))


# In[ ]:





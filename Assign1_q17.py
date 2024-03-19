#!/usr/bin/env python
# coding: utf-8

# In[7]:


#Code by: Yash Vardhan
import numpy as np

def eigen_value_qr_method(A,max_iter=100,tol=1e-6):
    V=np.eye(len(A))
    k=1
    while k<max_iter:
        Q,R=np.linalg.qr(A)
        A=np.dot(R,Q)
        V=np.dot(V,Q)
        off_diag_elements=A-np.diag(np.diag(A))
        if np.sum(np.abs(off_diag_elements))<tol:
            return np.diag(A),V,k
        k=k+1
    return np.diag(A),V,k
    
    
    

A=np.array([[5,-2],[-2,8]])
print("QR Decompostion method: ")
print("Q matrix: ",np.linalg.qr(A)[0])
print("R matrix: ",np.linalg.qr(A)[1])
print("Eigenvalues: ",eigen_value_qr_method(A)[0])
print("Matrix containing eigenvectors: ",eigen_value_qr_method(A)[1])
print("Eigenvalues (using inbuilt function) are: ",np.linalg.eigh(A)[0])


# In[ ]:





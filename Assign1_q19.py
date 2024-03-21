#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import time

def Smatrix_maker(mat,S):
    s_mat=np.zeros((np.shape(mat)))
    for i in range(np.shape(mat)[1]):
        s_mat[i][i]=S[i]
    return s_mat

def svd_decompose_time(matrix):
    # Measure time for SVD computation
    start_time = time.time()
    U, S, Vt = np.linalg.svd(matrix)
    end_time = time.time()

    # Print the results and time taken
    print("Original Matrix:")
    print(matrix)
    print("\nU Matrix:")
    print(U)
    print("\nSingular Values (S):")
    print(S)
    print("Matrix form of S: ")
    print(Smatrix_maker(matrix,S))
    print("\nVt Matrix:")
    print(Vt)
    print("Time Taken: ",(end_time - start_time))
    return U,Smatrix_maker(matrix,S),Vt


# part-a
A1=np.array([[2,1],[1,0]])
U1,S1,Vt1=svd_decompose_time(A1)
print("USVt:",np.dot(np.dot(U1,S1),Vt1))
print()
# part-b
A2=np.array([[2,1],[1,0],[0,1]])
U1,S1,Vt1=svd_decompose_time(A2)
print("USVt:",np.dot(np.dot(U1,S1),Vt1))
print()

# part-c
A3=np.array([[2,1],[-1,1],[1,1],[2,-1]])
U1,S1,Vt1=svd_decompose_time(A3)
print("USVt:",np.dot(np.dot(U1,S1),Vt1))
print()

# part-d
A4=np.array([[1,1,0],[-1,0,1],[0,1,-1],[1,1,-1]])
U1,S1,Vt1=svd_decompose_time(A4)
print("USVt:",np.dot(np.dot(U1,S1),Vt1))
print()

# part-e
A5=np.array([[1,1,0],[-1,0,1],[0,1,-1],[1,1,-1]])
U1,S1,Vt1=svd_decompose_time(A5)
print("USVt:",np.dot(np.dot(U1,S1),Vt1))
print()

# part-f
A6=np.array([[0,1,1],[0,1,0],[1,1,0],[0,1,0],[1,0,1]])
U1,S1,Vt1=svd_decompose_time(A6)
print("USVt:",np.dot(np.dot(U1,S1),Vt1))
print()



# In[ ]:





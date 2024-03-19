#!/usr/bin/env python
# coding: utf-8

# In[14]:


#CODE BY: Yash Vardhan
import numpy as np

def Jacobi_method2(A,b,X0=np.array([None]),tolerance=1e-2,max_iter=100,print_key=0,x_true=np.array([None])):
    '''
    A,b: Linear system of equation in the form Ax=b
    X0: Guess solution vector which would be used to start iterations
    tolerance: The value of precision we require in the code
    max_iter: maximum number of iterations for which the code will run
    print_key: if its value is non zero then steps will be printed 
    x_true: true solution if provided will be used for absolute error calculation and compared with tolerance
    '''
    if X0.any()==None:
        X0=np.zeros(len(b))  #if guess vector is not provided then the program will take null vector
     
    if x_true.any()==None:   #if true solution is not given then relative error will be calculated in the loop
        key=1
    else:
        key=0
        
    x_vec=np.zeros(len(b))   #initialising soln vector
    n = len(b)
    k=1                      #set k=1
    while k<=max_iter:       #running till iterations are less than or equal to max_iter
        
        if print_key!=0:       
            print()
            print("iteration: ",k)
            
        for i in range(n):       #executing jacobi algorithm
            s1=0
            for j in range(0,n):
                if j!=i:
                    s1=s1+A[i][j]*X0[j]
            x_vec[i] = (1/(A[i][i]))*(b[i]-s1)
        if key==1:               #calculation of relative error
            if (np.linalg.norm(x_vec-X0)/np.linalg.norm(x_vec))<tolerance:
                return x_vec,k
        else:
            if np.linalg.norm(x_vec - x_true) < tolerance:     #calculation of absolute error in case true soln is given
                return x_vec, k
            
        if print_key!=0:                    #print steps if I need to see them
            print("vector X0: ",X0)
            print("Vector x: ",x_vec)
         
        #updating k and X0
        k=k+1
        x_vec2=x_vec.copy()
        X0=x_vec2
        
    return x_vec,k

#System of equations
A = np.array([[0.2,0.1,1,1,0],[0.1,4,-1,1,-1],[1,-1,60,0,-2],[1,1,0,8,4],[0,-1,-2,4,700]])
b = np.array([1,2,3,4,5])

print(Jacobi_method2(A,b,tolerance=0.01,x_true=np.array([7.859713071,0.422926408,-0.073592239,-0.540643016,0.010626163])))



# In[11]:


def Gauss_Seidel_method2(A,b,X0=np.array([None]),tolerance=1e-2,max_iter=100,print_key=0,x_true=np.array([None])):
    '''
    A,b: Linear system of equation in the form Ax=b
    X0: Guess solution vector which would be used to start iterations
    tolerance: The value of precision we require in the code
    max_iter: maximum number of iterations for which the code will run
    print_key: if its value is non zero then steps will be printed 
    x_true: true solution if provided will be used for absolute error calculation and compared with tolerance
    '''    
    if X0.any()==None:
        X0=np.zeros(len(b))  #if guess vector is not provided then the program will take null vector
     
    if x_true.any()==None:   #if true solution is not given then relative error will be calculated in the loop
        key=1
    else:
        key=0
          
    
    x_vec=np.zeros(len(b))   
    n = len(b)
    k=1
    
    while k<=max_iter:
        
        if print_key!=0:
            print()
            print("iteration: ",k)
            print("X0: ",X0)
            
        for i in range(n):
            s1=0
            s2=0
            for j in range(0,i):
                s1=s1+A[i][j]*x_vec[j]
            for j2 in range(i+1,n):
                s2=s2+A[i][j2]*X0[j2]  
            sigma = s1+s2
            x_vec[i] =  (1 / A[i, i]) * (b[i] - sigma)
            if print_key!=0:
                print("x[",i,"]: ",x_vec[i])
        if key==1:
            if (np.linalg.norm(x_vec-X0)/np.linalg.norm(x_vec))<tolerance:
                return x_vec,k
        else:
            if np.linalg.norm(x_vec - x_true) < tolerance:
                return x_vec, k
        if print_key!=0:
            print("vector X0: ",X0)
            print("Vector x: ",x_vec)
    
         
        k=k+1
        x_vec2=x_vec.copy()
        X0=x_vec2
        
    return x_vec,k

#System of equations
A = np.array([[0.2,0.1,1,1,0],[0.1,4,-1,1,-1],[1,-1,60,0,-2],[1,1,0,8,4],[0,-1,-2,4,700]])
b = np.array([1,2,3,4,5])
print(Gauss_Seidel_method2(A,b,tolerance=0.01,x_true=np.array([7.859713071,0.422926408,-0.073592239,-0.540643016,0.010626163])))


# In[16]:


def relaxation_method2(A,b,omega=1.1,X0=np.array([None]),tolerance=1e-2,max_iter=100,print_key=0,x_true=np.array([None])):
    
    if X0.any()==None:
        X0=np.zeros(len(b))  #if guess vector is not provided then the program will take null vector
     
    if x_true.any()==None:   #if true solution is not given then relative error will be calculated in the loop
        key=1
    else:
        key=0
        
    x_vec=np.zeros(len(b))
    n = len(b)
    k=1
    
    while k<=max_iter:
        if print_key!=0:
            print()
            print("iteration: ",k)
            print("X0: ",X0)
        for i in range(n):
            s1=0
            s2=0
            for j in range(0,i):
                s1=s1+A[i][j]*x_vec[j]
            for j2 in range(i+1,n):
                s2=s2+A[i][j2]*X0[j2]  
            sigma = s1+s2
            x_vec[i] = (1-omega)*x_vec[i] + (omega / A[i, i]) * (b[i] - sigma)
            if print_key!=0:
                print("x[",i,"]: ",x_vec[i])
        if key==1:
            if (np.linalg.norm(x_vec-X0)/np.linalg.norm(x_vec))<tolerance:
                return x_vec,k
        else:
            if np.linalg.norm(x_vec - x_true) < tolerance:
                return x_vec, k
        if print_key!=0:
            print("vector X0: ",X0)
            print("Vector x: ",x_vec)
        
        k=k+1
        X0=x_vec.copy()
        
        
    return x_vec,k

A = np.array([[0.2,0.1,1,1,0],[0.1,4,-1,1,-1],[1,-1,60,0,-2],[1,1,0,8,4],[0,-1,-2,4,700]])
b = np.array([1,2,3,4,5])
print(relaxation_method2(A,b,omega=1.25,tolerance=0.01,x_true=np.array([7.859713071,0.422926408,-0.073592239,-0.540643016,0.010626163])))


# In[17]:


import numpy as np

def conjugate_gradient2(A,b,x0=None,max_iter=10**6,tol=0.01,x_true=None):    
    N=len(A)
    if x0==None:
        x=np.zeros(N)
    for k in range(max_iter):
        r=b-np.dot(A,x)
        t=(np.dot(r,r))/(np.dot(r,np.dot(A,r)))
        x=x+t*r
        if (np.linalg.norm(x-x_true))<tol:
            return x,k
    return x,k

A = np.array([[0.2,0.1,1,1,0],[0.1,4,-1,1,-1],[1,-1,60,0,-2],[1,1,0,8,4],[0,-1,-2,4,700]])
b = np.array([1,2,3,4,5])


print(conjugate_gradient2(A,b,x_true=np.array([7.859713071,0.422926408,-0.073592239,-0.540643016,0.010626163])))


# In[ ]:





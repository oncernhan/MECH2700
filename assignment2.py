#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 20:18:11 2018

@author: tanthanhnhanphan

MECH2700: Assignment 2
"""
import matplotlib.pyplot as plt
from numpy import *

from math import *

D = 100*10**3 #Dynamic Pressure
Ly = 1 #m
Lx = 5 #m
a = 20*pi/180
E = 70*10**9 #Young's Modulus
S = 96.5#Fatigue strength Mpa
Izz = 5*10**-5 #m4
ymax = 0.05 #m
FOS = 1.2


def q(i, n):
    load = D*Ly*sin(a)*(1-(i*Lx/n)**2/Lx**2)
    return load

"""
def x(n):
    for i in range(n+1):
        xx = i*Lx/n
        print(xx)
    
    

print(x(7))
"""
"""
A = array([[7, -4, 1, 0, 0, 0, 0],
            [-4, 6, -4, 1, 0, 0, 0],
            [1, -4, 6, -4, 1, 0, 0],
            [0, 1, -4, 6, -4, 1, 0],
            [0, 0, 1, -4, 6, -4, 1],
            [0, 0, 0, 1, -4, 5, -2],
            [0, 0, 0, 0, 2, -4, 2]], float)
"""    
#trial
"""
def s(n):
    space = Lx/n
    return space

h = s(7)
"""
#c = array([[q(1,7), q(2,7), q(3,7), q(4,7), q(5,7), q(6,7), q(7,7)]])*h**4/(E*Izz)

#b
#print(linspace(0,5,7))
def rhs(n):
    h= Lx/n
    q_1 = D*Ly*sin(a)*(1-(Lx/n)**2/Lx**2)*h**4/(E*Izz)
    load = array([[q_1]])
    #print(D*Ly*sin(a)*(1-(Lx/n)**2/Lx**2))
    for i in range(2, n+1):
        q_i = D*Ly*sin(a)*(1-(i*Lx/n)**2/Lx**2)*h**4/(E*Izz)
        #print(D*Ly*sin(a)*(1-(i*Lx/n)**2/Lx**2))
        #print(i*Lx/n)
        load = vstack((load, [[q_i]]))
    return load
"""
def rhs_1(n):
    h = Lx/n
    x = linspace(1,5,n)
    q_1 = D*Ly*sin(a)*(1-(x[1]**2/Lx**2)*h**4/(E*Izz))
    load = array([[q_1]])
    print(load)
    print(x)
    #for i in range(n+1):
    q_i = D*Ly*sin(a)*(1-x**2/Lx**2)*h**4/(E*Izz)
    print(q_i)
        #load = vstack((load, [[q_i]]))
        #load = vstack((load, q_i))
    return load
"""
        
#print("RHS",rhs(7))
#b = c.transpose()

def deflection(n):
    w = zeros((n,n))
    w[0,0] = 7
    w[n-2, n-2] = 5
    w[n-2, n-1] = -2  
    w[n-1, n-3] = 2
    w[n-1, n-1] = 2
    for k in range(0,n-1):
        w[k+1, k] = -4
    for k in range(0, n-3):
        w[k+1,k+1] = 6
    for k in range(0, n-2):
        w[k, k+2] = 1
    for k in range(0, n-3):
        w[k+2,k] = 1
    for k in range(0, n-2):
        w[k, k+1] = -4
    return w
    
#print(deflection((7)))
#print("~~~~")
#print(b)

#Direct Solver Gauss-Jordan Elimination



def solve(A,b, testmode = True):
    """
    Input: 
        A: nxn matrix of coefficients
        b: nx1 matrix of rhs values
    Output:
        x: solutions of Ax=b
        
    """
    nrows, ncols = A.shape
    c = hstack([A,b])
    #print(c)
    for j in range(0, nrows):
        p = j
        for i in range(j+1, nrows):
            #Select pivot
            if abs(c[i,j]) > abs(c[p,j]): p = i
        #Swap the rows
        c[p,:], c[j,:] = c[j,:].copy(), c[p,:].copy()
        #Elimination
        c[j,:] = c[j,:]/c[j,j]
        for i in range(0,nrows):
            if i!=j:
                c[i,:] = c[i,:] - c[i,j]*c[j,:]
    I, x = c[:,nrows], c[:,-1]

    return x
Alist = []
Blist = []
Clist = []
Dlist = []
Elist = []
rhslist = []
def solve_optimise(A,b):
    nrows, ncols = A.shape
    c = hstack([A,b])
    print(b)
    #b.tolist()
    #print(b)
    #print(c)
    for i in range(n-2):
        Alist.append(A[i, i+2])
    for i in range(n-1):
        Blist.append(A[i, i+1])
    for i in range(n):
        Clist.append(A[i,i])
    for i in range(n-1):
        Dlist.append(A[i+1, i])
    for i in range(n-2):
        Elist.append(A[i+2, i])
    for i in range(n):
        rhslist.append(b[i,0])
    rhslistcopy = rhslist.copy()
    """
    alpha = []
    mu = []
    gamma = []
    beta = []
    z = []
    mu_1 = Clist[0]
    alpha_1 =  Blist[0]/mu_1
    beta_1 = Alist[0]/mu_1
    z_1 = rhslist[0]/mu_1
    gamma_2 = Dlist[0]
    mu_2 = Clist[1] - alpha_1*gamma_2
    alpha_2 = (Blist[1]-beta_1*gamma_2)/mu_2
    beta_2 = Alist[1]/mu_2
    z_2 = (rhslist[1]-z_1*gamma_2)/mu_2
    alpha_minus2 = alpha_1
    alpha.append(alpha_1)
    alpha.append(alpha_2)
    mu.append(mu_1)
    mu.append(mu_2)
    gamma.append(gamma_2)
    beta.append(beta_1)
    z.append(z_1)
    z.append(z_2)
    print(gamma)
    for i in range(3, n-3):
        gamma_i = Dlist[i-2] - alpha[i-3]*Elist[i-3]
        mu_i = Clist[i-2] - beta[i-3]*Elist[i-3] - alpha[i-2]*gamma[i-2]
        beta_i = Alist[i-2]/mu_i
        gamma.append(gamma_i)
        
        beta.append(beta_i)
        z_i = (rhslist[i-1]-z[i-3])
    """    
    
        
    print(Alist)
    print(Blist)
    print(Clist)
    print(Dlist)
    print(Elist)
    print(rhslist)
    for i in range(n-1):
        multiplier_1 = Dlist[i]/Clist[i]
        #print('multi ',multiplier_1)
        #print(multiplier_1)
        #Dlist[i] = Dlist[i] - multiplier_1*Clist[i]
        #print('before ', rhslist[i+1])
        #rhslist[i+1] = rhslistcopy[i+1] - multiplier_1*rhslistcopy[i]
        #print('after', rhslist[i+1])
    #print(rhslist)
    #print(rhslistcopy)
    #print('~~~')    
    for i in range(n-2):
        multiplier_2 = Elist[i]/Clist[i]
        #print('multi ', multiplier_2)
        #print(multiplier_2)
        Elist[i] = Elist[i] - multiplier_2*Clist[i]
        #print('Before ',rhslist[i+2])
        #rhslist[i+2] = rhslist[i+2] - multiplier_2*rhslistcopy[i]
        #print('After ', rhslist[i+2])

    print(Dlist)
    print(Elist)
    
    #print(rhslist)
    #print(Clist[n-1])
    #x_n = rhslist[n-1]/Clist[n-1]
    #print(x_n)
    #for i in reversed(range(n)):
        #print(i)
    
    #for i in range(n-1):
        
        #print(multiplier_1)
    #print(Alist)
    return

solve_optimise(deflection(n), rhs(n))
#print(A)
#print(c)
#print(x)
#print(solve(deflection(280),rhs(280)))
for i in [7,14,28,280]:
    A = deflection(i)
    b = rhs(i)
    x = solve(A,b)
    xx= append(0, x)
    position = []
    for j in range(0,i+1):
        position.append(j*Lx/i)
    #print(position)
    plt.plot(position, xx, label=i)
    plt.xlabel('x(m)')
    plt.ylabel('Deflection (m)')
    plt.legend()
plt.show()

space = []
free_end_deflection = []
node = []

n = 280
#print(deflection(n))
#print(rhs(n))
sol = solve(deflection(n),rhs(n))
#print(sol)
sol_free_end = sol[[n-1]]
#print("Solution",sol_free_end)

for i in range(7, 50):
    A = deflection(i)
    b = rhs(i)
    x = solve(A,b)
    xx = x[[i-1]]
    free_end_deflection.append(xx)
    node.append(i)
    h = Lx/i
    space.append(h)
    if abs(xx - sol_free_end) < 0.1/100*sol_free_end:
        print(i)
        break
    
    #print(xx)
    #print(h)
#print(space)
plt.plot(space, free_end_deflection)
plt.show()


def moment_stress(n):
    A = deflection(n)
    b = rhs(n)
    x = solve(A,b)
    h = Lx/n
    #M_0 = E*Izz/(h**2)*(x[1]-2*x[0])
    M_1 = E*Izz/(h**2)*(x[1] - 2*x[0]) 
    #M = array([M_0])
    #M = hstack((M, [M_1]))
    M = array([M_1])
    #Stress
    #sigma_0 = M_0*ymax/Izz*10**-6
    sigma_1 = M_1*ymax/Izz*10**-6
    #sigma = array([sigma_0])
    #sigma = hstack((sigma, sigma_1))
    sigma = array([sigma_1])
    #print(M)
    for i in range(1,n-1):
        M_i = E*Izz/(h**2)*(x[[i+1]]- 2*x[i] + x[i-1])
        sigma_i = M_i*ymax/Izz*10**-6
        #print(M_i)
        M = hstack((M, M_i))
        sigma = hstack((sigma, sigma_i))
        #load = vstack((load, [[q_i]]))
    M = hstack((M, [0]))
    sigma = hstack((sigma, [0]))
    
    
    position = []
    for j in range(1, n+1):
        position.append(j*Lx/n)
    print(max(sigma))
    #print(sigma[1])
    print(len(position))
    print(len(M))
    Izz_new = max(M)*ymax*FOS/(S*10**6)*10**5
    print("Izz hey baby cum at me",Izz_new)
    plt.plot(position, M)
    plt.title('Bending moment vs. length')
    plt.xlabel('x (m)')
    plt.ylabel('Bending moment (Nm)')
    plt.show()
    
    plt.plot(position, sigma)
    plt.title('Bending stress vs. length')
    plt.xlabel('x (m)')
    plt.ylabel('Bending stress (MPa)')
    plt.show()
    
    #print(M)
        
    return 

moment_stress(17)

#print('RHS: ',rhs(7))
#print('RHS 1: "',rhs_1(7))
#print(rhs(7))
"""
#First line of matrix
import time 
A = deflection(n)
b = rhs(n)

##################Computation time using optimised solver######################

start_op = time.time()

deflect_op = solve(A, b)

end_op = time.time()
compute_time_op = end_op - start_op
print("Computation time using optimised solver:", compute_time_op)

###############Computation time using numpy in-built solver#####################

start = time.time()

deflect = np.linalg.solve(A, b)

end = time.time()
compute_time = end - start
print("Computation time using numpy in-built solver:", compute_time)

###############Computation time using Gauss-Jordan solver#######################

start_g = time.time()

deflect_g = solve(A, b)

end_g = time.time()
compute_time_g = end_g - start_g

print("Computation time using Gauss-Jordan solver:", compute_time_g)



print("Does the solver work? \n", check(deflect_op, deflect))
"""














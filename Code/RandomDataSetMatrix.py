# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 21:04:41 2022

@author: ESIOT
"""

import numpy as np
import matplotlib.pyplot as plt
import ot
from sklearn.metrics.pairwise import pairwise_distances
import scipy.optimize
import seaborn as sns
from Matcher import Matcher
from utils import KL
from train_parameters import train_parameters
from model_parameters import model_parameters

sns.set_style("darkgrid", {'axes.grid' : False})



outer_iteration = 50
inner_iteration = 20
learning_rate = 1

seed = 0
rng = np.random.RandomState(seed)

# Initialise the users and items
m, n = 20, 20
n_sample = 50 * m * n
p, q, r = 10, 8, 4

# Initialise probability distributions
U = rng.randn(p, m)
V = rng.rand(q, n)
G0 = rng.rand(r, p)
D0 = rng.rand(r, q)
A0 = np.dot(G0.T, D0)
gamma0, const0, degree0 = 0.1, 1, 1
C0 = np.power(gamma0 * U.T.dot(A0).dot(V) + const0, degree0)
ax = sns.heatmap(C0) 
r0, c0 = np.abs(rng.randn(m)), np.abs(rng.randn(n))
r0 /= r0.sum()
c0 /= c0.sum()

pi0 = ot.rot(C0, r0, c0)[0]

C1 = 5 * pairwise_distances(rng.randn(m, 2))
C2 = 5 * pairwise_distances(rng.randn(n, 2))

eps, eps1, eps2 = 1, 1, 1

def rel_error(M, M0):
    return np.linalg.norm(M - M0) / np.linalg.norm(M0)

def loss(pi, pi_sample, reg_para):
    ans = -np.sum(pi_sample * np.log(pi)) \
        + reg_para * (ot.rot(C1, pi.sum(axis=1), pi_sample.sum(axis=1))[-1] + ot.rot(C2, pi.sum(axis=0), pi_sample.sum(axis=0))[-1])
    return ans
noise = 0.08
pi_sample = pi0 + noise * np.abs(rng.randn(m, n))
pi_sample /= pi_sample.sum()

     
G = np.eye(r, p)
D = np.eye(r, q)
A = np.dot(G.T, D)
            

gamma, const, degree = 0.05, 1, 2
C = np.power(gamma * U.T.dot(A).dot(V) + const, degree)
r_sample, c_sample = pi_sample.sum(axis=1), pi_sample.sum(axis=0)
v = eps1 * np.log(ot.rot(C1, r_sample, r_sample, lam=1/eps1)[2])
w = eps2 * np.log(ot.rot(C2, c_sample, c_sample, lam=1/eps2)[2])
v_dual = eps1*np.log(r_sample) -eps1 * np.log(np.sum(np.exp((np.outer(np.ones(m), v) - C1) / eps1), axis=0))
w_dual = eps2*np.log(c_sample) -eps2 * np.log(np.sum(np.exp((np.outer(np.ones(n), w) - C2) / eps2), axis=0))

pi, xi, eta= ot.rot(C, r_sample, c_sample, eps)[:-1]


losses = np.zeros(outer_iteration)
KLs = np.zeros(outer_iteration)
constraints = np.zeros((outer_iteration, 3))
best_loss = np.inf
best_configuration = None

            
for i in range(outer_iteration):
    Z = np.exp(-C / eps)
    M = 0.001 * (np.outer(v, np.ones(n)) + np.outer(np.ones(m), w)) * Z 
  
    for j in range(inner_iteration):                    
        def f1(lam):
            xi1 = (r_sample / (M - lam * Z).dot(eta))
            return xi1.dot(Z).dot(eta) - 1
        
        def f2(lam):
            return xi.dot(Z).dot(c_sample / (M - lam * Z).T.dot(xi)) - 1
        
        lam0 = np.min(M.dot(eta) / Z.dot(eta))
        lam = scipy.optimize.root(f1, lam0-10).x[0]
        xi = r_sample / (M - lam * Z).dot(eta)

        lam0 = np.min(M.dot(eta) / Z.dot(eta))
        lam = scipy.optimize.root(f2, lam0-10).x[0]
        eta = c_sample / (M - lam * Z).T.dot(xi)
   

    constraint1 = np.linalg.norm(-r_sample / xi + M.dot(eta) - lam * Z.dot(eta))
    constraint2 = np.linalg.norm(-c_sample / eta + M.T.dot(xi) - lam * Z.T.dot(xi))
    constraint3 = xi.dot(Z).dot(eta) - 1
    constraints[i] = [constraint1, constraint2, constraint3]
    pi = np.dot(np.diag(xi), np.exp(-C / eps) * eta)
    
    if constraint1 > 1e-8 or constraint2 > 1e-8 or constraint3 > 1e-8:
        break

    grad_C = (pi_sample + (lam - 0.001*(np.outer(v, np.ones(n)) + np.outer(np.ones(m), w))) * pi) / eps

    factor = grad_C * degree * gamma * np.power(gamma * U.T.dot(A).dot(V) + const , degree - 1)
    grad_A = U.dot(factor).dot(V.T) + 0.00 * A
    A -= learning_rate * grad_A
    C = np.power(gamma * U.T.dot(A).dot(V) + const, degree)

    
    v = eps1 * np.log(ot.rot(C1, pi.sum(axis=1), r_sample, eps1)[2])
    w = eps2 * np.log(ot.rot(C2, pi.sum(axis=0), c_sample, eps2)[2])
    v_dual = eps1*np.log(pi.sum(axis=1)) -eps1 * np.log(np.sum(np.exp((np.outer(np.ones(m), v) - C1) / eps1), axis=0))
    w_dual = eps2*np.log(pi.sum(axis=0)) -eps2 * np.log(np.sum(np.exp((np.outer(np.ones(n), w) - C2) / eps2), axis=0))
    
    losses[i] = loss(pi, pi_sample, 0.001)
    KLs[i] = KL(pi, pi_sample)
    
    if KLs[i] < best_loss:
        best_configuration = [A, C, pi]

        
model = Matcher(pi_sample, U, V)

train_param = train_parameters(max_outer_iteration=500, max_inner_iteration=20, learning_rate=500)
model_param = model_parameters(A0=A0+rng.randn(p, q), gamma=0.02, const=1, degree=2, lam=1.0, lambda_mu=1.0, lambda_nu=1.0, delta=0.005)

C_old, A_old = model.polynomial_kernel(model_param, train_param)[:2]
pi_old = ot.rot(C_old, r_sample, c_sample)[0]

# Plot matrixes

plt.clf() #release memory
plt.close() #kill the figure
ax = sns.heatmap(C0)
plt.savefig("initialCostMatrix.png")
#ax.savefig() 

plt.clf() #release memory
plt.close() #kill the figure
ax7 = sns.heatmap(pi0)
plt.savefig("initialPiMatrix.png") 

plt.clf() #release memory
plt.close() #kill the figure
ax1 = sns.heatmap(C_old)
plt.savefig("otCostMatrix.png")    

 
plt.clf() #release memory
plt.close() #kill the figure
ax3 = sns.heatmap(C)
plt.savefig("riotCostMatrix.png")

plt.clf() #release memory
plt.close() #kill the figure
ax4 = sns.heatmap(pi_old)
plt.savefig("otPiMatrix.png")    

plt.clf() #release memory
plt.close() #kill the figure
ax6 = sns.heatmap(pi)
plt.savefig("riotPiMatrix.png")

plt.clf() #release memory
plt.close() #kill the figure
ax6 = sns.heatmap(pi_sample)
plt.savefig("optPiMatrix.png")
   

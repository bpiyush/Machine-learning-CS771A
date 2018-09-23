import pandas as pd
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det
from scipy.optimize import optimize
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
from sympy import *
from sympy.plotting.plot import List2DSeries

x, y, z, t = symbols('x y z t')

m1 = np.array([10,20])
s1 = np.array([2,1])

m2 = np.array([20,10])
s2 = np.array([1,-1])
D = 2

x1 = [i for i in range(-5,35)]
y1 = x1
plt.ion()
file = '{}.png'.format('hello')
hp = plot_implicit(Eq(s1[0]*(x - m1[0])**2 + s1[1]*(y - m1[0])**2 - s2[0]*(x - m2[0])**2 - s2[1]*(y - m2[1])**2, 0),(x, -5, 35), (y, -5, 35))
fig = hp._backend.fig
ax = hp._backend.ax
xx = yy = np.linspace(-5,35)
ax.plot(xx,yy)
ax.plot([0],[0],'o') # Point (0,0)
ax.set_aspect('equal','datalim')
plt.show(block=True)


# P = List2DSeries(Eq(s1[0]*(x - m1[0])**2 + s1[1]*(y - m1[0])**2 - s2[0]*(x - m2[0])**2 - s2[1]*(y - m2[1])**2, 0),(x,-5,35))
# h = P.get_points()
# print(type(h))
# print(h[0])

# fig.show()
# graph.append(List2DSeries(x1, y1))
# graph.save(file)

# ax = plt.figure()
# plt.plot(x1,y1)

# file = '{}.png'.format('hello2')
# graph = plot_implicit(Eq(s1[0]*(x - m1[0])**2 + s1[1]*(y - m1[0])**2 - s2[0]*(x - m2[0])**2 - s2[1]*(y - m2[1])**2, 0),(x, -5, 35), (y, -5, 35))
# graph.append(ax)
# graph.save(file)


# def boundary(Z):

# 	X1 = np.power((X - m1), 2)
# 	X2 = np.power((X - m2), 2)
# 	Y1 = np.power(((Y.T - m1.T).T), 2)
# 	Y2 = np.power(((Y.T - m2.T).T), 2)
# 	return s1[0]*X1-Y1*s2[0] + s1[1]*X2-Y2*s2[1]


# X, Y =  optimize.fsolve(boundary, (1, 1))

# x = np.linspace(-5,35,500)
# y = np.linspace(-5,35,500)
# X, Y = np.meshgrid(x,y)
# pos = np.empty(X.shape + (2,))
# pos[:, :, 0] = X; pos[:, :, 1] = Y
# # print(pos.shape, pos[:,:,0].shape)
# plt.figure(figsize=(10,12))
# Z = (pos - m1)@s1@(pos-m1) - (pos - m2)@s2@(pos-m2)
# plt.contour(X,Y,Z,[0])

# fig = plt.figure()
# hc=plt.contourf(X,Y,boundary(X,Y)[0].clip(-1,1),levels=np.linspace(-1,1,20))
# plt.contour(X,Y,boundary(X,Y)[0].clip(-1,1),levels=[0],color=(1,0,0))
# plt.colorbar(hc)

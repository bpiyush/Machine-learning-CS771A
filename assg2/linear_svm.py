import pandas as pd
import numpy as np
from numpy.linalg import inv
from sympy import *
from sympy.plotting.plot import List2DSeries
from numpy.linalg import det
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
from sklearn.svm import LinearSVC
import sys


# Load the data as pandas dataframe
if len(sys.argv) > 1:
	if sys.argv[1] == '-1':
		# Primary dataset
		data = pd.read_csv('binclass.txt', delimiter=',',header=None)
	elif sys.argv[1] == '-2':
		# Secondary dataset
		data = pd.read_csv('binclassv2.txt', delimiter=',',header=None)
else:
	sys.exit("Please see the README file to run the code correctly. You need to give arguments after the usual commands.")

# Convert it into numpy ndarray
data = data.values
# print(data.shape)

# Input examples
X = data[:,:-1]
D = X.shape[1]
# Labels
Y = data[:,-1]
# print(X.shape, Y.shape)

# Seperate the positive class
X_pos = X[np.where(Y==1)]

# Seperate the negative class
X_neg = X[np.where(Y==-1)]
# print(X_pos.shape, X_neg.shape)

model = LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, 
	intercept_scaling=1, class_weight=None, verbose=0, random_state=0, max_iter=5000)
model.fit(X, Y)

# print(model.coef_, model.intercept_)

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = -15,43
    y_min, y_max = -32,32
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contour(xx, yy, Z, **params)
    return out


# Set-up 2x2 grid for plotting.

plt.figure(figsize=(15,12))
ax= plt.gca()
plt.title('Linear SVM: Decision boundary')
plt.xlabel('Input Dimension 1')
plt.ylabel('Input Dimension 2')
ax.set_xlim([-15,43])
ax.set_ylim([-32,32])
ax.scatter(X_pos[:,0], X_pos[:,1], marker='+', c='r', s=[35 for i in range(X_pos.shape[0])], label='Positive')
ax.scatter(X_neg[:,0], X_neg[:,1], marker='+', c='b', s=[35 for i in range(X_pos.shape[0])], label='Negative')
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)
plot_contours(ax, model, xx, yy,
              cmap=plt.cm.coolwarm, alpha=0.8)
plt.legend()
plt.show()
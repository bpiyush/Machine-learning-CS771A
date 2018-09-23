import pandas as pd
import numpy as np
from numpy.linalg import inv
from sympy import *
from sympy.plotting.plot import List2DSeries
from numpy.linalg import det
from scipy.optimize import minimize
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
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

# Seperate the positive class
X_pos = X[np.where(Y==1)]

# Seperate the negative class
X_neg = X[np.where(Y==-1)]

# Go to the consolidated plot function and start reading the code from there.

def gaussian(theta,x):
	"""
	Computes log[N(x | mu, [sigma_1, sigma_2, .., sigma_D])] where N denotes gaussian distribution with parameters mu, sigma
	Args:
		theta(array): set of parameters [mu, sigma]
	Returns:
		float
	""" 

	mu = theta[:D]
	# sigma = np.diag(theta[D:])
	delta = 0.0001 # To avoid division by zero error
	sigma = ((theta[D]**2)+delta)*np.identity(D)
	# Dimension
	sigma_inv = inv(sigma)
	term = (-0.5)*((x-mu)@ sigma_inv @ (x-mu)) # @ denotes matrix multiplication
	term = np.exp(term)
	constant = 1/(np.power((2*np.pi), D)*det(sigma))

	if constant*term !=0:
		return -np.log(constant*term)
	else:
		return -np.log(constant*term + 0.00001)


def likelihood(theta, data):
	"""
	Returns the final likelihood by summing up individual terms obtained from gaussian(). 
	"""
	sum_term = 0
	for i in range(data.shape[0]):
		sum_term = sum_term + gaussian(theta,data[i,:])
	return sum_term


def mle_solution(X, string):
	"""
	Solves the optimization problem to find the mle estimate for given data and returns the optimal parameters.
	"""
	# Initialization: mu_i = 0.8 for all i in {1,2,..,D} 
	mu_start = np.array([0.8 for i in range(D)])
	# Initialization: sigma = 1.5
	sigma_start = np.array([1.5])

	print("Fitting the likelihood minimzing model for "+ string + ":")
	model = minimize(fun=likelihood, x0=np.concatenate((mu_start, sigma_start)), args=X, method='L-BFGS-B',options={'disp': False})

	# Extracting the optimal parameters
	mu_opt = model.x[:D]
	sigma_opt = ((model.x[D])**2)*np.identity(D)

	return mu_opt, sigma_opt


def likelihood_with_given_covariance(theta, data):
	"""
	Returns the final likelihood by summing up individual terms obtained from gaussian(). 
	"""
	sigma = data['sigma']
	data = data['X']
	sum_term = 0
	for i in range(data.shape[0]):
		sum_term = sum_term + gaussian_with_given_covariance(theta,data[i,:], sigma)
	return sum_term

def gaussian_with_given_covariance(theta,x, sigma):
	"""
	Computes log[N(x | mu, [sigma_1, sigma_2, .., sigma_D])] where N denotes gaussian distribution with parameters mu, sigma
	Args:
		theta(array): set of parameters [mu, sigma]
	Returns:
		float
	""" 

	mu = theta
	# sigma = np.diag(theta[D:])
	delta = 0.0001 # To avoid division by zero error
	sigma = ((sigma**2)+delta)*np.identity(D)
	# Dimension
	sigma_inv = inv(sigma)
	term = (-0.5)*((x-mu)@ sigma_inv @ (x-mu)) # @ denotes matrix multiplication
	term = np.exp(term)
	constant = 1/(np.power((2*np.pi), D)*det(sigma))

	if constant*term !=0:
		return -np.log(constant*term)
	else:
		return -np.log(constant*term + 0.00001)

def mle_solution_with_given_covariance(X, sigma, string):
	"""
	Solves the optimization problem to find the mle estimate for mean for given variance.
	"""
	# Initialization: mu_i = 0.8 for all i in {1,2,..,D} 
	mu_start = np.array([0.8 for i in range(D)])

	print("Fitting the likelihood minimzing model for "+ string+" :")
	additional = {'X':X, 'sigma':sigma}
	model = minimize(fun=likelihood_with_given_covariance, x0=mu_start, args=additional, method='L-BFGS-B',options={'disp': False})

	# Extracting the optimal parameters
	mu_opt = model.x

	return mu_opt


def initialize_plot():
	"""
	Initilizes a template for plot
	"""
	plt.figure(figsize=(15,12))
	plt.title('Class conditional probability distributions: Contours')
	plt.xlabel('Input Dimension 1')
	plt.ylabel('Input Dimension 2')


def consolidated_plot(X_pos, X_neg, same_cov=0):
	"""
	Main function that computes and plots the requirements.
	Arguments:
		X_pos (ndarray): Set of positive examples
		X_neg (ndarray): Set of negative examples
	Returns:
		None: Computes mu's and sigma's using optimization and plots the required things.
	"""
	initialize_plot()
	ax = plt.gca()

	# Compute the meshgrid using numpy
	x = np.linspace(-5,35,500)
	y = np.linspace(-5,35,500)
	X, Y = np.meshgrid(x,y)
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X; pos[:, :, 1] = Y

	# Get the solutions from optimization problems solved for positive and negative classes
	mu_1, sigma_1 = mle_solution(X_pos, 'positive class with different covariances: ++++')
	mu_2, sigma_2 = mle_solution(X_neg, 'negative class with different covariances: ----')


	# Plot the positive examples and contour of gaussian for positive examples
	normal1 = multivariate_normal(mu_1, sigma_1)
	plt.plot(X_pos[:,0],X_pos[:,1], 'r+', label='Positive')
	plt.contour(X, Y, normal1.pdf(pos), 15, cmap=plt.cm.OrRd)

	# Plot the negative examples and contour of gaussian for negative examples
	normal2 = multivariate_normal(mu_2, sigma_2)
	plt.plot(X_neg[:,0],X_neg[:,1], 'b+', label='Negative')
	plt.contour(X, Y, normal2.pdf(pos), 10, cmap=plt.cm.PuBu)

	# Plot the decision boundary
	p = normal1.pdf(pos) - normal2.pdf(pos)
	plt.contour(X, Y, p, levels=[0])

	plt.legend()
	plt.show()
	pass


def consolidated_plot_with_given_covariance(X_pos, X_neg, same_cov=0):
	"""
	Main function that computes and plots the requirements.
	Arguments:
		X_pos (ndarray): Set of positive examples
		X_neg (ndarray): Set of negative examples
	Returns:
		None: Computes mu's and sigma's using optimization and plots the required things.
	"""
	initialize_plot()
	ax = plt.gca()

	# Compute the meshgrid using numpy
	x = np.linspace(-20,35,500)
	y = np.linspace(-32,35,500)
	X, Y = np.meshgrid(x,y)
	pos = np.empty(X.shape + (2,))
	pos[:, :, 0] = X; pos[:, :, 1] = Y

	# Get the solutions from optimization problems solved for positive and negative classes
	mu_1, sigma_1 = mle_solution(X_pos, 'positive class with same covariances: ++++')

	# For the same variance case for negative class
	mu_2 = mle_solution_with_given_covariance(X_neg, sigma_1, 'negative class with same covariances: ----')
	sigma_2 = sigma_1

	# Plot the positive examples and contour of gaussian for positive examples
	normal1 = multivariate_normal(mu_1, sigma_1)
	plt.plot(X_pos[:,0],X_pos[:,1], 'r+', label='Positive')
	plt.contour(X, Y, normal1.pdf(pos), 15, cmap=plt.cm.OrRd)

	# Plot the negative examples and contour of gaussian for negative examples
	normal2 = multivariate_normal(mu_2, sigma_2)
	plt.plot(X_neg[:,0],X_neg[:,1], 'b+', label='Negative')
	plt.contour(X, Y, normal2.pdf(pos), 10, cmap=plt.cm.PuBu)

	# Plot the decision boundary
	p = normal1.pdf(pos) - normal2.pdf(pos)
	plt.contour(X, Y, p, levels=[0])

	plt.legend()
	plt.show()
	pass

consolidated_plot(X_pos, X_neg)

consolidated_plot_with_given_covariance(X_pos, X_neg)

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# from sklearn.kernel_ridge import KernelRidge # Implemented from scratch but also used library to compare performance
from sklearn.metrics import mean_squared_error 
# from sklearn.linear_model import Ridge # Implemented from scratch but also used library to compare performance

train = pd.read_csv('ridgetrain.txt', delimiter='\n')
train = train.values
train = [[float(y) for y in x[0].split()] for x in train]

test = pd.read_csv('ridgetest.txt', delimiter=',')
test = test.values
test = [[float(y) for y in x[0].split()] for x in test]


x_train = np.array([x[0] for x in train]).reshape((-1,1))
y_train = [x[1] for x in train]

x_test = np.array([x[0] for x in test]).reshape((-1,1))
y_test = [x[1] for x in test]


def rbf_kernel(x,y, gamma):
	return np.exp(-1.0*gamma*((np.linalg.norm(x-y))**2))

def kernel_matrix(X1, X2, gamma):
	"""
	Returns the kernel matrix for dataset X1 onto X2 of shape NxD
	In other words, Returns [k(x^{(1)}_n, x^{(2)}_m)]
	"""
	t = np.zeros((X1.shape[0], X2.shape[0]))
	for i in range(X1.shape[0]):
		for j in range(X2.shape[0]):
			t[i][j] = rbf_kernel(X1[i,:], X2[j,:], gamma)
	return t

def compute_alpha(X,y, ridge_lambda, gamma):
	y = np.array(y).reshape((-1,1))
	mat = kernel_matrix(X, X, gamma) + ridge_lambda*np.identity(X.shape[0])
	return np.dot(np.linalg.inv(mat),y)

def kernel_ridge_predict(x_train, x_test, y_train, ridge_lambda, gamma):
	alpha = compute_alpha(x_train, y_train, ridge_lambda, gamma)
	kernel_test = kernel_matrix(x_test, x_train, gamma)
	prediction = np.dot(kernel_test,alpha)
	return prediction
	
# def kernel_ridge_predict_using_library(x_train, x_test, y_train, ridge_lambda, gamma):
# 	KR = KernelRidge(alpha=1, kernel='rbf', gamma=gamma)
# 	KR.fit(x_train, y_train)
# 	return KR.predict(x_test)

def kernel_ridge_regression(x_train, x_test, y_train, y_test, ridge_lambda, gamma):
	y_predicted = kernel_ridge_predict(x_train, x_test, y_train, ridge_lambda, gamma)
	y_test = np.array(y_test).reshape((-1,1))

	x_test = [x[0] for x in x_test]
	y_test = [x[0] for x in y_test]
	y_predicted = [x[0] for x in y_predicted]

	# Computing MSE
	mse = mean_squared_error(y_test, y_predicted)
	print("The mean squared error for kernel ridge regression for lambda = ",ridge_lambda, " is ", mse)

	# Plotting the data and results
	plt.figure(figsize=(15,12))
	plt.title("Kernel Ridge Regression with lambda = " + str(ridge_lambda))
	plt.xlabel("x-axis")
	plt.ylabel("y-axis")
	plt.scatter(x_test, y_test, c="b", label="True labels")
	plt.scatter(x_test, y_predicted, c="r", label="Predicted labels")
	plt.legend()
	plt.show()

	
# Running the kernel ridge regression model
lambda_set = [0.1,1, 10, 100]
for x in lambda_set:
	kernel_ridge_regression(x_train, x_test, y_train, y_test, x, 0.1)

def transform(data, landmark_data, gamma):
	return kernel_matrix(data, landmark_data, gamma)

def landmark_ridge(x_train, x_test, y_train, y_test, ridge_lambda, gamma, L):
	# landmarks = random.sample(range(1, x_train.shape[0]), L)
	landmarks = [1,5,6]
	landmark_data = x_train[landmarks]
	data = np.concatenate((x_train, x_test), axis=0)
	low_dim_data = transform(data, landmark_data, gamma)

	x_train_new = low_dim_data[:x_train.shape[0],:]
	x_test_new = low_dim_data[x_train.shape[0]:,:]

	y_predicted = kernel_ridge_predict(x_train_new, x_test_new, y_train, ridge_lambda, gamma)

	# Computing MSE
	mse = mean_squared_error(y_test, y_predicted)
	print("The mean squared error for landmark-based ridge regression for L = ",L, " is ", mse)

	# Plotting the data and results
	plt.figure(figsize=(15,12))
	plt.title("Landmark-based Ridge Regression with L = " + str(L))
	plt.xlabel("x-axis")
	plt.ylabel("y-axis")
	plt.scatter(x_test, y_test, c="b", label="True labels")
	plt.scatter(x_test, y_predicted, c="r", label="Predicted labels")
	plt.legend()
	plt.show()

# Running the landmark-ridge regression
L_set = [2, 5, 20, 50, 100]
for L in L_set:
	landmark_ridge(x_train, x_test, y_train, y_test, 0.1, 0.1, L)



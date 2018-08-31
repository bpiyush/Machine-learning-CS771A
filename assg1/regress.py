import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.multioutput  import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


def prototype_classification_predict(means, X_test, Y_test, method, lambda1):
	'''
	Runs a prototype based classification on data (X_test, Y_test) with given means.
	Args:
		means (np array): means of the classes for prototype based classification
		X_test (np ndarray): test input set
		Y_test (np array): test output labels
	'''
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(means)
	distances, indices = nbrs.kneighbors(X_test)
	indices = indices + 1.0
	print("Test accuracy for prototype based classification with "+ method + " and lambda = "+ str(lambda1) + ": ", accuracy_score(Y_test, indices))
	return accuracy_score(Y_test, indices)


if __name__=="__main__":

	# List of class attribute vector for seen classes
	class_attributes_seen = np.load('AwA_python/class_attributes_seen.npy')
	# List of class attribute vector for unseen classes
	class_attributes_unseen = np.load('AwA_python/class_attributes_unseen.npy')
	# Loading the seen input data
	X_seen = np.load('AwA_python/X_seen.npy',encoding='bytes')
	seen_means = []
	# Populating the list of seen means
	for x in range(40):
		seen_means.append(np.mean(X_seen[x],axis=0))
	seen_means = np.array(seen_means)

	# Code to train a linear model to predict the unseen means
	X = class_attributes_seen
	Y = seen_means
	
	# Values that lambda can take
	lambdas = [0.01, 0.1, 1, 10, 20, 50, 100]

	# List to store accuracies
	accuracies = []
	
	# For loop to run the regression model for each value of lambda
	for lambda1 in lambdas:
		# Fit the multi-output linear regression model
		regressor = MultiOutputRegressor(Ridge(alpha=lambda1))
		regressor.fit(X, Y)
	
		# Use the trained multi-output regressor to get the unseen mean values
		unseen_means = regressor.predict(class_attributes_unseen)
	
		# Load the test datasets
		X_test = np.load('AwA_python/Xtest.npy')
		Y_test = np.load('AwA_python/Ytest.npy')
	
		# Run the prototype based classification for method 2
		acc = prototype_classification_predict(unseen_means, X_test, Y_test, 'method 2', lambda1)
		accuracies.append(acc)
	'''
	# Plot accuracies vs lambdas
	plt.figure()
	ax = plt.gca()
	ax.set_xticks([0.01, 0.1, 1, 10, 20, 50, 100])
	plt.plot(lambdas, accuracies)
	plt.title('Test accuracy vs Lambda')
	plt.xlabel('Lambda')
	plt.ylabel('Test accuracy')
	plt.grid()
	plt.show()
	'''

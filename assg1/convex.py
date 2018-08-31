import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

def prototype_classification_predict(means, X_test, Y_test, method):
	'''
	Runs a prototype based classification on data (X_test, Y_test) with given means.
	Args:
		means (np array): means of the classes for prototype based classification
		X_test (np ndarray): test input set
		Y_test (np array): test output labels
	'''
	nbrs = NearestNeighbors(n_neighbors=1, leaf_size=10,algorithm='kd_tree', metric='minkowski', p=1.5).fit(means)
	distances, indices = nbrs.kneighbors(X_test)
	indices = indices + 1.0
	print("Test accuracy for prototype based classification with "+ method + ": ", accuracy_score(Y_test, indices))


if __name__ == "__main__":
	
	# List of class attribute vector for seen classes
	class_attributes_seen = np.load('AwA_python/class_attributes_seen.npy')

	# List of class attribute vector for unseen classes
	class_attributes_unseen = np.load('AwA_python/class_attributes_unseen.npy')

	# Matrix containing similarity coefficients
	similarity_matrix = np.inner(class_attributes_unseen, class_attributes_seen)

	# Normalizing each row by dividing by sum of row
	similarity_matrix = (similarity_matrix.T/np.sum(similarity_matrix, axis=1)).T

	# Loading the seen input data
	X_seen = np.load('AwA_python/X_seen.npy',encoding='bytes')

	seen_means = []
	# Populating the list of seen means
	for x in range(40):
		seen_means.append(np.mean(X_seen[x],axis=0))
	seen_means = np.array(seen_means)

	# Finding the unseen means
	unseen_means = np.dot(similarity_matrix, seen_means)

	# Load the test datasets
	X_test = np.load('AwA_python/Xtest.npy')
	Y_test = np.load('AwA_python/Ytest.npy')

	# Run the prototype based classification for method 1
	prototype_classification_predict(unseen_means, X_test, Y_test, 'method 1')
   
	

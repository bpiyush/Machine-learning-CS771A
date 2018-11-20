import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

data = pd.read_csv('kmeans_data.txt', delimiter=',')
data = data.values

X1_list = [float(x[0].split()[0]) for x in data]
X2_list = [float(x[0].split()[1]) for x in data]
X1 = np.array(X1_list).reshape((data.shape[0],-1))
X2 = np.array(X2_list).reshape((data.shape[0],-1))

data = np.concatenate((X1,X2), axis=1)
print("Shape of dataset: ", data.shape)

# Visualize data 
plt.figure(figsize=(15,12))
plt.title("Dataset for k-means clustering")
plt.xlabel("x-axis")
plt.ylabel("y-axis")
plt.plot(X1_list, X2_list, "+b", label="Original dataset")
plt.legend()
plt.show()
plt.close()

def euclidean(a, b):
    return np.linalg.norm(a - b)

def kmeans_clustering(dataset, k, initial_centers, num_iter):
	d = dataset.shape[1]
	n = dataset.shape[0]
	centers = initial_centers
	assignments = [0 for x in range(n)]

	for s in range(num_iter):
		for i in range(n):
			distances = [euclidean(dataset[i,:], centers[j]) for j in range(k)]
			assignments[i] = np.argmin(distances)

		for j in range(k):
			centers[j] = np.mean(dataset[np.where(np.array(assignments)==j)], axis=0)
	return np.array(assignments)


def clustering_with_feature_transformation(data):
	# I am using the transformed feature space as (x1, x2, x1^2+x2^2)
	X1 = data[:,0].reshape((-1,1))
	X2 = data[:,1].reshape((-1,1))
	X3 = X1**2+X2**2
	X3 = X3.reshape((-1,1))
	print(X1.shape, X3.shape)
	transformed_data = np.concatenate((X1, X2, X3),axis=1)

	initial_centers = transformed_data[:2,:]

	############## Tried with sk learn k means to verify my results #################
	# model = KMeans(n_clusters=2, init=initial_centers, n_init=1).fit(transformed_data)
	# labels_predicted = model.labels_
	# idx= labels_predicted

	idx2 = kmeans_clustering(transformed_data, 2, initial_centers, 20)

	plt.figure(figsize=(15,12))
	plt.title("Clustering on transformed data in the original space: K = 2")
	plt.xlabel("x-axis")
	plt.ylabel("y-axis")
	plt.plot(data[idx2==0,0],data[idx2==0,1],'+r', data[idx2==1,0],data[idx2==1,1],'+g')
	plt.show()

clustering_with_feature_transformation(data)

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

def landmark_clustering(data, gamma, L=1):
	landmark = random.sample(range(1, data.shape[0]), L)
	landmark_data = data[landmark]
	low_dim_data = kernel_matrix(data, landmark_data, gamma)

	initial_centers = low_dim_data[:2,:]


	############## Tried with sk learn k means to verify my results #################
	# model = KMeans(n_clusters=2, init=initial_centers, n_init=1).fit(low_dim_data)
	# labels_predicted = model.labels_
	# idx= labels_predicted

	idx = kmeans_clustering(low_dim_data, 2, initial_centers, 20)

	plt.figure(figsize=(15,12))
	plt.title("Landmark based Clustering: K = 2")
	plt.xlabel("x-axis")
	plt.ylabel("y-axis")
	# print(landmark)
	plt.plot(data[idx==0,0],data[idx==0,1],'+r', data[idx==1,0],data[idx==1,1],'+g')
	plt.scatter(data[landmark[0],0], data[landmark[0],1], c='b', label="Landmark point", s=[100])
	plt.legend()
	plt.show()

for i in range(10):
	landmark_clustering(data, gamma=0.1)

import pickle
import numpy as np
import pandas as pd
import tqdm
from scipy import ndimage
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

with open('data/facedata_py2_3.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']

def alt_opt_ppca(X, K):
    N = X.shape[0]
    D = X.shape[1]
    
    # Initialize the params and latent variables
    W = np.random.rand(D, K)
    Z = np.random.rand(N, K)
    
    for iteration in tqdm.trange(100):
        Z = (np.dot(np.dot(np.linalg.inv(np.dot(W.T, W)), W.T), X.T)).T
        W = (np.dot(np.dot(np.linalg.inv(np.dot(Z.T, Z)), Z.T), X)).T
    
    return Z, W

def plot_images(X_original, X_reconstructed,k):
    from scipy import ndimage

    L = []
    for i in range(5):
        L.append(X_original[i])
    for i in range(5):
        L.append(X_reconstructed[i])

    fig=plt.figure(figsize=(12, 5))
    columns = 5
    rows = 2

    for i in range(1, columns*rows +1):
        img = L[i-1].reshape(64, 64)
        img = ndimage.rotate(img, 270)
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
#     plt.savefig('img_k_'+str(k)+'.png')
    plt.show()
    
    return L

def plot_weights(W, k, l=10):
    W_subset = W[:,:10]
    
    fig=plt.figure(figsize=(12, 5))
    columns = 5
    rows = 2

    for i in range(1, columns*rows +1):
        img = W_subset[:,i-1].reshape(64, 64)
        img = ndimage.rotate(img, 270)
        fig.add_subplot(rows, columns, i)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
#     plt.savefig('template_weights_k_'+str(k)+'.png')
    plt.show()
    
    return W_subset

def test():
    mu = np.mean(X, axis=0)
    X_centered = X-mu


    k_values = [10, 20,30, 40, 50, 100]

    all_images = []
    all_basis = []

    for k in k_values:
        print("For k = ", k)
        Z, W = alt_opt_ppca(X_centered, k)

        reconstruct_indices = [10, 20, 30, 40, 50]

        X_original = X[reconstruct_indices,:]
        X_reconstructed = mu + np.dot((Z[reconstruct_indices,:]), W.T)

        scaler = MinMaxScaler(feature_range=(0,255))
        X_reconstructed = (scaler.fit_transform(X_reconstructed.T)).T

        L = plot_images(X_original, X_reconstructed, k)
        W_subset = plot_weights(W, k)

        all_images.append(L)
        all_basis.append(W_subset)

# Run function 
test()
    
    

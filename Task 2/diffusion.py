import numpy as np

from scipy.spatial import KDTree
from scipy.sparse import diags
from scipy.sparse.linalg import inv, eigs

import matplotlib.pyplot as plt

def diffusion_map(X, L = 5, max_dist_threshold = None):
    """Performs diffusion mapping on an array
    
    Parameters
    ----------
    X : np.ndarray
        the array to be deconstructed
    L : int
        number of eigen values to be computed
    max_dist_threshold : float
        An estimate of the data's diameter, i.e the max distance
        between two points
        
    Returns
    -------
    eigen_values : np.ndarray
        largest eigen values found, sorted by magnitude
    eigen_vectors : np.ndarray
        associated eigen vectors
    
    """
    if max_dist_threshold is None:
        # estimates the data's diameter as its upper bound 
        max_dist_threshold = np.max(np.abs(X)) * np.sqrt(X.shape[-1])
        print(f"Estimated data's diamater: {max_dist_threshold:.3f}")
        
    # Creates neighboor grid
    X_t = KDTree(X)
    
    eps = max_dist_threshold * .05

    # Computes distance matrix as a sparce matrix using an array representation
    D = X_t.sparse_distance_matrix(
        X_t, 
        3*eps, # multiplied by 3 to ensure 99.7% of relevant information is taken into account
        output_type='coo_matrix'
    )

    # Computes exp(-D²/e) - 1 (so it doesn't affect zeros)
    W = (-D.multiply(D)/eps).expm1()
    
    # removes the "-1" to get exp(-D²/e)
    #W.eliminate_zeros()
    W.data+=1

    # Computes P and P^-1. The sum returns a matrix, flatten to get a 1D list to form a diagonal matrix
    P = diags(W.sum(axis=1).getA1())
    P_inv = inv(P)

    K = P_inv @ W @ P_inv

    # Computes Q^-1/2
    Q = diags(K.sum(axis=1).getA1())
    Q_sqrt_inv = inv(Q).sqrt()

    T = Q_sqrt_inv @ K @ Q_sqrt_inv

    # Compute eigenvalues and vectors
    w, u = eigs(T, k=L+1, which='LM')

    eigen_vectors = Q_sqrt_inv @ u
    eigen_values = np.power(w, 1/(2*eps))

    return eigen_values, eigen_vectors

def diff_plt(vectors, lambdas, against, ignore_first_n = 2, plot_on_T = False, c = "b"):
    """plot the given vectors against another one

    Parameters
    -------
        vectors : np.ndarray 
            eigen vectors to plot, of shape n_samples, n_components
        lambdas : np.ndarray
            associated eigen values
        against : np.ndarray
            the vector to plot against, of shape n_samples
        ignore_first_n : int
            number of vectors to ignore in vectors
        plot_on_T : bool (opt.)
            The against is a time serie
        c : np.ndarray
            Color of each points
            
    Returns
    -------
        None
    """
    
    vectors = vectors[:, ignore_first_n:]
    lambdas = lambdas[ignore_first_n:]
    _, n_vectors = vectors.shape
    
    n_row = 1 + (n_vectors - 1) // 5

    _, axs = plt.subplots(n_row, 5, figsize=(20, 5 * n_row), sharex=True, sharey=True)
    
    # in case we have <=5 vectors to plot
    if n_row == 1:
        for j in range(5):
            if j >= n_vectors:
                break
            axs[j].scatter(against, vectors[:, j ], s=50, c=c)
            if plot_on_T:
                axs[j].set_title(f"T vs \u03C8_{j + ignore_first_n}, \u03BB: {lambdas[j].real:.3f}")
            else:
                axs[j].set_title(f"\u03C8_1 vs \u03C8_{j + ignore_first_n}, \u03BB: {lambdas[j].real:.3f}")
            #axs[j].set_ylim([-1, 1])
        return
    
    
    for i in range(n_row):
        for j in range(5):
            if i*5 + j >= n_vectors:
                break
            axs[i, j].scatter(against, vectors[:, i*5 + j ], s=50, c=c)
            if plot_on_T:
                axs[i, j].set_title(f"T vs \u03C8_{i*5 + j + ignore_first_n}, \u03BB: {lambdas[i*5 + j].real:.3f}")
            else:
                axs[i, j].set_title(f"\u03C8_1 vs \u03C8_{i*5 + j + ignore_first_n}, \u03BB: {lambdas[i*5 + j].real:.3f}")
            #axs[i, j].set_ylim([-1, 1])

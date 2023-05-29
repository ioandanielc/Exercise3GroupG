import numpy as np

def perform_PCA(array):
    """Performs PCA on an array

        Parameters
        ----------
        array : np.ndarray
            the array to be deconstructed

        Returns
        -------
        center : np.ndarray
            center of data_task_1
        centered_array : np.ndarray
            centered data_task_1
        U, sigma, Vh :  np.ndarray
            the components of the SVD Decomposition
        Sigma : np.ndarray
            diagonal matrix of sigma for a later reconstruction of data_task_1
        energy : np.ndarray
            energy per PC
        explained_variance -: np.ndarray
            explained_variance
        """

    # calculate the center
    center = np.average(array, axis=0)
    # center the matrix
    centered_array = array - center

    # Perform the SVD decomposition
    U, sigma, Vh = np.linalg.svd(centered_array, full_matrices=True)

    energy = np.square(sigma)

    #Calculate the trace
    trace = sum(np.square(sigma))


    # Calculating the explained variance
    explained_variance = energy.copy()
    for i in range(explained_variance.shape[0]):
        explained_variance[i] = sum(energy[0:i + 1])/trace

    # Calculating the amount of energy per principal component
    energy = np.square(sigma)/np.sum(np.square(sigma))

    # Create a diagonal matrix from sigma that can be used to create truncated data_task_1
    Sigma = np.zeros((U.shape[0], Vh.shape[0]))
    for i in range(min(U.shape[0], Vh.shape[0])):
        Sigma[i, i] = sigma[i]

    return center, centered_array, U, sigma, Sigma, Vh, energy, explained_variance


def reconstruct_PCA(U, Sigma, Vh):
    """Reconstructs an array given the SVD decomposition

        Parameters
        ----------
        U, Sigma, Vh : np.ndarray
            the components of the SVD Decomposition

        Returns
        -------
        array : np.ndarray
            the new array
        """
    reconstruction_array = U @ Sigma @ Vh
    return reconstruction_array

def truncate(U, Sigma, Vh, a):
    """Reconstructs an array given the SVD decomposition

        Parameters
        ----------
        U, Sigma, Vh : np.ndarray
            the components of the SVD Decomposition
        a : int
            only the firs a PC-s are kept

        Returns
        -------
        reconstruction_array : np.ndarray
            the new array
        """

    # Keeping only the first "a" principal components
    new_Sigma = Sigma.copy()

    # If we do not want to remove any of the PC-s, we will just
    # recreate the matrix
    if a == Sigma.shape[1]:
        reconstruction_array = U @ Sigma @ Vh
        return reconstruction_array

    # a has to be a positive integer
    elif a <= 0:
        raise SystemError("a has to be  apositive integer")

    # set the columns corresponding to the removed PC-s to 0
    new_Sigma[:, a:Sigma.shape[1]] = np.zeros((Sigma.shape[0], Sigma.shape[1] - a))

    # Creating the new data_task_1
    reconstruction_array = U @ new_Sigma @ Vh

    return reconstruction_array

def recenter_data(array, center):
    """Recenters an array given its previous center

        Parameters
        ----------
        array : np.ndarray
            the array to be centered
        center : np.ndarray
            the previous center

        Returns
        -------
        centered_array : np.ndarray
            centered data_task_1
        """
    centered_array = array + center
    return centered_array

def plot_paths(plt, new_numbers):
    """Plots the path of every pedestrian

        Parameters
        ----------
        plt : matplotlib.pyplot
        new_numbers : np.ndarray
            the paths after the PCA transformation

        Returns
        -------
        """

    # Every pedestrian has two coordinates per line, we use the
    # start variable to be able to always jump 2 entries, when we are changing
    # the pedestrian.
    start = 0
    pedestrians = range(15)
    for ped in pedestrians:
        steps_of_ped = new_numbers[:, start:(start + 2)]
        plt.scatter(steps_of_ped[:, 0], steps_of_ped[:, 1])
        start += 2
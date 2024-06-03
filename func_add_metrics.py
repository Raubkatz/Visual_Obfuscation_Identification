import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
import random
import numpy as np
import random
from differint.differint import GL
from collections import Counter
from math import log
from numpy.linalg import matrix_rank
from sklearn.utils import shuffle
from copy import deepcopy as dc


def singular_value_energy(singular_values, k=3):
    """
    Calculate the energy of the dominant singular values.

    Parameters:
    singular_values (array-like): Array of singular values from SVD.
    k (int, optional): Number of dominant singular values to consider. Default is 3.

    Returns:
    float: The ratio of the sum of squares of the first k singular values 
           to the sum of squares of all singular values.
    """
    dominant_sum_square = sum([val**2 for val in singular_values[:k]])
    total_sum_square = sum([val**2 for val in singular_values])
    return dominant_sum_square / total_sum_square


def relative_decay_vals(singular_values):
    """
    Calculate the decay values of singular values.

    Parameters:
    singular_values (array-like): Array of singular values from SVD.

    Returns:
    list: A list of differences between consecutive singular values.
    """
    decay = [singular_values[i] - singular_values[i+1] for i in range(len(singular_values)-1)]
    return decay


def relative_decay(singular_values):
    """
    Calculate the average decay of singular values.

    Parameters:
    singular_values (array-like): Array of singular values from SVD.

    Returns:
    float: The average difference between consecutive singular values.
    """
    decay = [singular_values[i] - singular_values[i+1] for i in range(len(singular_values)-1)]
    average_decay = sum(decay) / len(decay)
    return average_decay


def calculate_fisher_information(matrix, random_scramble=0, make_square=False, valid=True):
    """
    Calculate the Fisher Information of singular values of a matrix.

    Parameters:
    matrix (numpy.ndarray): Input matrix.
    random_scramble (int, optional): Controls the type of scrambling.
        0 - no scrambling (default)
        1 - alternate row and column scrambling
        2 - row scrambling only
        3 - column scrambling only
    make_square (bool, optional): If True, convert the matrix to a square matrix first. Default is False.
    valid (bool, optional): If True, filter out singular values below a threshold. Default is True.

    Returns:
    float: The Fisher Information calculated from the singular values.
    """
    if make_square:
        if matrix.shape[0] < matrix.shape[1]:
            matrix = matrix @ matrix.T
        else:
            matrix = matrix.T @ matrix

    if random_scramble > 0:
        fisher_infos = []
        for i in range(1000):
            # generate random number of scrambles
            n_scrambles = random.randint(1, 100)

            # scramble rows and columns
            for j in range(n_scrambles):
                if random_scramble == 1:
                    if i % 2 == 0:  # even iterations
                        # swap two random columns
                        col_idxs = np.random.choice(matrix.shape[1], size=2, replace=False)
                        matrix[:, [col_idxs[0], col_idxs[1]]] = matrix[:, [col_idxs[1], col_idxs[0]]]
                    else:  # odd iterations
                        # swap two random rows
                        row_idxs = np.random.choice(matrix.shape[0], size=2, replace=False)
                        matrix[[row_idxs[0], row_idxs[1]], :] = matrix[[row_idxs[1], row_idxs[0]], :]
                elif random_scramble == 2:
                    # swap two random rows
                    row_idxs = np.random.choice(matrix.shape[0], size=2, replace=False)
                    matrix[[row_idxs[0], row_idxs[1]], :] = matrix[[row_idxs[1], row_idxs[0]], :]
                elif random_scramble == 3:
                    # swap two random columns
                    col_idxs = np.random.choice(matrix.shape[1], size=2, replace=False)
                    matrix[:, [col_idxs[0], col_idxs[1]]] = matrix[:, [col_idxs[1], col_idxs[0]]]

            singular_values = np.linalg.svd(matrix, compute_uv=False)
            singular_values /= np.sum(singular_values)  # normalize singular values
            if valid:
                threshold = 0.000001
                singular_values = dc(singular_values[singular_values > threshold])
            FI = (singular_values[1:] - singular_values[:-1]) ** 2 / singular_values[:-1]
            fisher_infos.append(np.sum(FI))  # we sum the FI to get a single value for the matrix

        # return average Fisher Information
        return sum(fisher_infos) / len(fisher_infos)

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if valid:
        threshold = 0.00001
        singular_values = dc(singular_values[singular_values > threshold])
    singular_values /= np.sum(singular_values)  # normalize singular values
    FI = (singular_values[1:] - singular_values[:-1]) ** 2 / singular_values[:-1]

    return np.sum(FI)  # we sum the FI to get a single value for the matrix


def calculate_entropy(matrix, random_scramble=0, make_square=False, valid=True):
    """
    Calculate the entropy of singular values of a matrix.

    Parameters:
    matrix (numpy.ndarray): Input matrix.
    random_scramble (int, optional): Controls the type of scrambling.
        0 - no scrambling (default)
        1 - alternate row and column scrambling
        2 - row scrambling only
        3 - column scrambling only
    make_square (bool, optional): If True, convert the matrix to a square matrix first. Default is False.
    valid (bool, optional): If True, filter out singular values below a threshold. Default is True.

    Returns:
    float: The entropy of the singular values.
    """
    if make_square:
        if matrix.shape[0] < matrix.shape[1]:
            matrix = matrix @ matrix.T
        else:
            matrix = matrix.T @ matrix

    if random_scramble > 0:
        entropies = []
        for i in range(1000):
            # generate random number of scrambles
            n_scrambles = random.randint(1, 100)

            # scramble rows and columns
            for j in range(n_scrambles):
                if random_scramble == 1:
                    if i % 2 == 0:  # even iterations
                        # swap two random columns
                        col_idxs = np.random.choice(matrix.shape[1], size=2, replace=False)
                        matrix[:, [col_idxs[0], col_idxs[1]]] = matrix[:, [col_idxs[1], col_idxs[0]]]
                    else:  # odd iterations
                        # swap two random rows
                        row_idxs = np.random.choice(matrix.shape[0], size=2, replace=False)
                        matrix[[row_idxs[0], row_idxs[1]], :] = matrix[[row_idxs[1], row_idxs[0]], :]
                elif random_scramble == 2:
                    # swap two random rows
                    row_idxs = np.random.choice(matrix.shape[0], size=2, replace=False)
                    matrix[[row_idxs[0], row_idxs[1]], :] = matrix[[row_idxs[1], row_idxs[0]], :]
                elif random_scramble == 3:
                    # swap two random columns
                    col_idxs = np.random.choice(matrix.shape[1], size=2, replace=False)
                    matrix[:, [col_idxs[0], col_idxs[1]]] = matrix[:, [col_idxs[1], col_idxs[0]]]

            singular_values = np.linalg.svd(matrix, compute_uv=False)
            if valid:
                threshold = 0.000001
                singular_values = dc(singular_values[singular_values > threshold])
            sv_entropy = entropy(singular_values)
            entropies.append(sv_entropy)

        # return average entropy
        return sum(entropies) / len(entropies)

    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if valid:
        threshold = 0.000001
        singular_values = dc(singular_values[singular_values > threshold])
    sv_entropy = entropy(singular_values)

    return sv_entropy



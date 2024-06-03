"""
This script processes matrices /greyscale images stored in pickle files. It calculates several complexity
metrics based on Singular Value Decomposition (SVD) and saves these metrics along with class information into a CSV file.
The code also includes options for grouping class information based on specific criteria.

Author: Sebastian Raubitzek
"""

import os
import numpy as np
import pandas as pd
import pickle
from copy import deepcopy as dc
import re
from func_add_metrics import calculate_entropy, calculate_fisher_information, relative_decay, singular_value_energy

# Set Parameters #######################################################################################################
grouping = 0  # 0: no grouping, 1: binary grouping, 2: type and no obfuscation grouping, 3: no obfuscation and all Tigress types
valid = True  # Needs to be true in order to keep only singular values that are above a certain threshold
data_directory = "./image_data"
########################################################################################################################

# List to hold all matrix data
data_list = []

# Loop over each file in the directory
for file in os.listdir(data_directory):
    file_path = os.path.join(data_directory, file)

    # Check if it's a file
    if os.path.isfile(file_path):
        print(f"Processing {file}")

        # Load the pickle file containing the matrix
        with open(file_path, 'rb') as f:
            matrix = pickle.load(f)
            matrix = np.array(matrix)  # Ensure it's a numpy array

        # Extract class information from the file title
        first_underscore = file.find('_') + 1
        last_underscore = file.rfind('_')
        if first_underscore > 0 and last_underscore > first_underscore:
            class_info = file[first_underscore:last_underscore]
            # Remove optimization level and digits
            class_info = re.sub(r'O\d+$', '', class_info)
        else:
            class_info = "Unknown"

        # Special class handling based on keywords
        if grouping != 0:
            if grouping < 3:
                if "tigress" in class_info:
                    if "encode" in class_info:
                        class_info = "TigressDataObfuscation"
                    elif "flatten" in class_info or "split" in class_info:
                        class_info = "TigressCFGObfuscation"
                    elif "virtualize" in class_info or "jit" in class_info:
                        class_info = "TigressDynamicObfuscation"
                if grouping == 1:
                    if "Tigress" in class_info:
                        class_info = "TigressObfuscation"
            else:
                if "tigress" in class_info:
                    if "encodeArithmetic" in class_info:
                        class_info = "TigressEncodeArithmetic"
                    elif "encodeLiterals" in class_info:
                        class_info = "TigressEncodeLiterals"
                    elif "flatten" in class_info:
                        class_info = "TigressFlatten"
                    elif "split" in class_info:
                        class_info = "TigressSplit"
                    elif "virtualize" in class_info:
                        class_info = "TigressVirtualize"
                    elif "jit" in class_info:
                        class_info = "TigressJit"
            if grouping >= 1:
                if "oslatest" in class_info or "tinycc" in class_info or "tendra-latest" in class_info:
                    class_info = "NoObfuscation"
        else:
            if class_info[-1] == "_" or class_info[-1] == "-":
                class_info = class_info[:-1]

        print(class_info)

        # Create a sample 2D array for visualization
        array = np.random.randint(0, 256, np.shape(matrix), dtype=np.uint8)

        # Calculate various SVD-based metrics
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        if valid:
            threshold = 0.000001
            singular_values = dc(singular_values[singular_values > threshold])
        svd_entropy = calculate_entropy(matrix, valid=True)
        singular_spectral_radius = max(abs(singular_values))

        if singular_values.size > 0:
            svd_condition_number = np.max(singular_values) / np.min(singular_values)
        else:
            svd_condition_number = 0  # Handle case with no valid min singular value

        svd_relative_decay = relative_decay(singular_values)
        svd_energy = singular_value_energy(singular_values)
        svd_fisher_info = calculate_fisher_information(matrix, valid=True)

        # Append the processed data to the list if not excluded
        matrix_data = {
            'svd_entropy': svd_entropy,
            'singular_spectral_radius': singular_spectral_radius,
            'svd_condition_number': svd_condition_number,
            'svd_relative_decay': svd_relative_decay,
            'svd_energy': svd_energy,
            'svd_fisher_info': svd_fisher_info,
            'class': class_info
        }

        data_list.append(matrix_data)

# Convert list to DataFrame
df = pd.DataFrame(data_list)
print(df.head())  # Print the first few rows to check

# Save DataFrame to a CSV file
df.to_csv(f'final_processed_matrices_grouping{grouping}_ext.csv', index=False)

"""
This script leverages the ExtraTreesClassifier for predictive modeling, coupled with comprehensive data preparation, model evaluation, and interpretability frameworks. It automates the end-to-end process of fitting, evaluating, and interpreting ensemble tree-based machine learning models on complex datasets.

Author: Sebastian Raubitzek

Features:
- **Data Loading and Preprocessing**: Automates the ingestion and preliminary preprocessing of datasets, ensuring they are fit for analysis and modeling.
- **Model Training with Hyperparameter Optimization**: Utilizes ExtraTreesClassifier, enhanced with Bayesian optimization (via BayesSearchCV) for hyperparameter tuning, aiming to optimize model performance.
- **Model Evaluation**: Employs a suite of metrics (accuracy, precision, recall, F1 score, confusion matrix) for a thorough evaluation of model performance, complemented by detailed classification reports.
- **Feature Importance Analysis**: Analyzes and visualizes the importance of each feature in the prediction process, providing insights into the dataset's underlying structure and influence on model decisions.
- **SHAP Value Interpretation**: Integrates SHAP (SHapley Additive exPlanations) for model interpretation, offering detailed insights into the contribution of each feature towards individual predictions.
- **Result Visualization and Storage**: Visualizes key metrics and SHAP values for interpretability, stores model artifacts, evaluation metrics, and plots for further analysis.

Workflow:
1. The script initializes by defining a parameter grid for the ExtraTreesClassifier, setting up model training and evaluation configurations.
2. It proceeds to load training and testing datasets, preparing them for the modeling process.
3. The ExtraTreesClassifier model is then trained with Bayesian optimization for hyperparameter tuning, ensuring optimal model configuration.
4. Post-training, the model is evaluated using various metrics, and a classification report is generated for a comprehensive performance overview.
5. Feature importance is calculated and visualized to understand feature contributions to the model's predictions.
6. Finally, all results, including performance metrics, feature importance, and SHAP summaries, are saved for documentation and review.

Usage:
- Ensure all dependencies are installed and datasets are located at the specified paths.
- Configure the PARAMETERS section according to the specific requirements of the dataset and predictive task at hand.
- Execute the script in an environment that supports the utilized libraries for seamless operation.
"""

from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from skopt import BayesSearchCV
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import ADASYN

warnings.filterwarnings("ignore")

# Define the parameter grid for hyperparameter optimization of the ExtraTreesClassifier
parameter_grid_extratrees = {
    'n_estimators': [80, 100, 120, 140, 160, 180, 200],
    'criterion': ['gini', 'entropy'],
    'max_features': ['log2', 'sqrt', None],
    'max_depth': list(range(10, 51, 10)),  # Example: from 10 to 50, stepping by 10
    'min_samples_split': [2, 4, 6, 8, 10],
    'min_samples_leaf': [1, 2, 3, 4, 5]
}

def perform_oversampling(X_train, y_train, oversampling, specific_sample_count):
    """
    Applies oversampling to the training data using the ADASYN algorithm to mitigate class imbalance.

    Parameters:
    - X_train (DataFrame or ndarray): The training feature set.
    - y_train (Series or ndarray): The target values for the training set.
    - oversampling (bool): Flag indicating whether oversampling should be applied.
    - specific_sample_count (int): The target number of samples for each class after oversampling.

    Returns:
    - X_resampled, y_resampled (tuple): The resampled feature set and target values, respectively. If oversampling is not applied, returns the original datasets.
    """
    if not oversampling:
        return X_train, y_train
    oversampler = ADASYN(random_state=137, sampling_strategy={cls: specific_sample_count for cls in np.unique(y_train)})
    return oversampler.fit_resample(X_train, y_train)

########################################################################################################################
# PARAMETERS ###########################################################################################################
########################################################################################################################
oversampling = True  # Determines whether to apply oversampling to balance the class distribution.
specific_sample_count = 3500  # The target number of instances per class after applying oversampling.
seed = 137  # Seed for random number generators to ensure reproducibility.
grouping = 1  # Selector for the target variable: `0` for `y3`, `1` for `y4`, `2` for `y5`.
opt_n_iter = 300  # Number of iterations for hyperparameter optimization
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Define folder paths for saving results and models based on parameterization
folder_name = "./Results_ML"
trained_models_dir = f"{folder_name}/trained_models"
results_output_dir = f"{folder_name}/results"

# Create necessary directories if they do not exist
os.makedirs(trained_models_dir, exist_ok=True)
os.makedirs(results_output_dir, exist_ok=True)

# File path for storing classification metrics
output_file = f"{folder_name}/classification_metrics_extratrees_grouping{grouping}.txt"

# Load the dataset
data = pd.read_csv(f'./ML_data/processed_matrices_grouping{grouping}.csv')

# Preprocess the data
X = data.drop('class', axis=1)
y = data['class']

# Output the size of the dataset
print(f"Data Dimensions: {data.shape[0]} rows, {data.shape[1]} columns")
print(f"Features (X) Dimensions: {X.shape[0]} rows, {X.shape[1]} columns")
print(f"Target (y) Dimensions: {y.shape[0]} rows")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Perform oversampling on the training set if enabled to address class imbalance
if oversampling:
    print('Oversampling')
    X_train, y_train = perform_oversampling(X_train, y_train, oversampling, specific_sample_count)

print("Data loading and preparation complete.")

# Initialize the ExtraTreesClassifier model
model = ExtraTreesClassifier()

# Start hyperparameter optimization using Bayesian optimization
start_time_hp = time.time()
bocv = BayesSearchCV(model, parameter_grid_extratrees, n_iter=opt_n_iter, cv=5, verbose=1)
bocv.fit(X_train, y_train)
end_time_hp = time.time()
hp_search_time = end_time_hp - start_time_hp

# Use the best estimator from hyperparameter optimization
model = bocv.best_estimator_
best_params = bocv.best_params_
best_cv_score = bocv.best_score_

# Train the model with the best parameters
start_time_train = time.time()
model.fit(X_train, y_train)
end_time_train = time.time()
training_time = end_time_train - start_time_train

# Make predictions on the test set
start_time_pred = time.time()
y_pred = model.predict(X_test)
end_time_pred = time.time()
prediction_time = end_time_pred - start_time_pred

# Calculate classification metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate and print the classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Calculate feature importances
feature_importances = model.feature_importances_
feature_names = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Store metrics and model details
results = {
    "best_params": best_params,
    "best_cv_score": best_cv_score,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "confusion_matrix": conf_matrix.tolist(),
    "hp_search_time": hp_search_time,
    "training_time": training_time,
    "prediction_time": prediction_time,
    "feature_importances": feature_importance_df.to_dict('records')  # Convert dataframe to list of dicts
}

# Print and save the metrics
metrics_summary = f"""
#######################################################################################################
Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}

Timing Metrics:
Hyperparameter Search Time: {hp_search_time:.2f} seconds
Training Time: {training_time:.2f} seconds
Prediction Time: {prediction_time:.2f} seconds
Total Time: {hp_search_time + training_time + prediction_time:.2f} seconds

Best CV Score: {best_cv_score:.4f}

"""
metrics_summary += report

# Append the feature importances to the metrics summary
metrics_summary += "\nFeature Importances (Normalized):\n"
for index, row in feature_importance_df.iterrows():
    metrics_summary += f"{row['Feature']}: {row['Importance']:.4f}\n"

print(metrics_summary)
with open(output_file, "a") as file:
    file.write(metrics_summary)

# Save the results and model
results_pickle_file = os.path.join(results_output_dir, f"classification_results_extratrees_grouping{grouping}.pickle")
with open(results_pickle_file, 'wb') as handle:
    pickle.dump(results, handle)

model_filename = os.path.join(trained_models_dir, f"best_extratrees_model_grouping{grouping}.pkl")
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

# Visualize and save the confusion matrix
class_names = np.unique(y_train)
conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
categories = [x.replace('prog_', '') for x in sorted(np.unique(y_test))]

plt.figure(figsize=(20, 16), dpi=100)
sns.set(font_scale=1.2)
g = sns.heatmap(conf_matrix_normalized, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, xticklabels=categories, yticklabels=categories)

for xtick, color in zip(g.axes.get_xticklabels(), ['black'] * len(categories)):
    xtick.set_color(color)
for ytick, color in zip(g.axes.get_yticklabels(), ['black'] * len(categories)):
    ytick.set_color(color)

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title("Normalized Confusion Matrix ExtraTrees")
plt.tight_layout()
plt.savefig(os.path.join(results_output_dir, f"normalized_confusion_matrix_grouping{grouping}.png"), dpi=100)
plt.savefig(os.path.join(results_output_dir, f"normalized_confusion_matrix_grouping{grouping}.eps"), format='eps')
plt.clf()
plt.close()

# Visualize and save the feature importances
plt.figure(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title('ExtraTrees Feature Importances')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig(os.path.join(results_output_dir, f"feature_importances_extratrees_grouping{grouping}.png"))
plt.savefig(os.path.join(results_output_dir, f"feature_importances_extratrees_grouping{grouping}.eps"))
plt.clf()
plt.close()

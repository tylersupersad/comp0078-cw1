import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from question4 import (
    split_data_set,
    load_data,
    baseline_regression,
    linear_regression_single_attribute,
    linear_regression_all_attributes,
)

# compute the gaussian kernel between two vectors
def gaussian_kernel(vector1, vector2, sigma):
    # calculate the difference between the two vectors
    difference = vector1 - vector2
    # return the gaussian kernel value
    return np.exp(-np.dot(difference, difference) / (2 * sigma**2))

# compute the kernel matrix for a dataset using the gaussian kernel
def compute_kernel_matrix(data, sigma):
    num_samples = len(data)
    # initialize the kernel matrix
    kernel_matrix = np.zeros((num_samples, num_samples))
    
    for row in range(num_samples):
        for col in range(num_samples):
            # compute the gaussian kernel value for each pair of samples
            kernel_matrix[row, col] = gaussian_kernel(data[row], data[col], sigma)
            
    return kernel_matrix

# perform kernel ridge regression to compute alpha values
def kernel_ridge_regression(kernel_matrix, targets, regularization_param):
    # number of training samples
    num_samples = len(targets) 
    # create an identity matrix for regularization
    identity_matrix = np.eye(num_samples)  
    # solve for alpha
    return np.linalg.solve(kernel_matrix + regularization_param * identity_matrix, targets)

# compute the kernel matrix between training and test datasets
# this is used to evaluate the model on unseen data
def compute_test_kernel_matrix(training_data, test_data, sigma):
    # generate a kernel matrix where each entry is the similarity between a test and training sample
    return np.array([[gaussian_kernel(train_point, test_point, sigma) for test_point in training_data] for train_point in test_data])

# perform 5-fold cross-validation to find the best gamma and sigma
def five_fold_cross_validation(data, gamma_values, sigma_values):
    # shuffle the data for randomness in the folds
    np.random.shuffle(data) 
    # split the data into 5 equal parts 
    folds = np.array_split(data, 5) 
    # initialize the best parameters
    best_gamma, best_sigma, lowest_mse = None, None, float('inf') 
    # to store mse for all gamma and sigma combinations
    mse_results = [] 
    
    for gamma in gamma_values:
        for sigma in sigma_values:
            # to store mse for each fold
            fold_mse_list = []  
            
            for fold_index in range(5):
                # split data into training and validation sets for the current fold
                validation_data = folds[fold_index]
                training_data = np.vstack([folds[i] for i in range(5) if i != fold_index])

                # separate features and targets
                train_features = training_data[:, :-1]
                train_targets = training_data[:, -1]
                validation_features = validation_data[:, :-1]
                validation_targets = validation_data[:, -1]

                # compute kernel matrix for training data
                training_kernel_matrix = compute_kernel_matrix(train_features, sigma)
                alpha_values = kernel_ridge_regression(training_kernel_matrix, train_targets, gamma)

                # compute kernel matrix for validation data
                validation_kernel_matrix = compute_test_kernel_matrix(train_features, validation_features, sigma)
                predictions = predict(validation_kernel_matrix, alpha_values)

                # calculate mse for the validation set
                mse = np.mean((predictions - validation_targets) ** 2)
                fold_mse_list.append(mse)

            # calculate average mse across all folds for this gamma and sigma
            avg_mse = np.mean(fold_mse_list)
            mse_results.append((gamma, sigma, avg_mse))

            # update the best parameters if a lower mse is found
            if avg_mse < lowest_mse:
                best_gamma, best_sigma, lowest_mse = gamma, sigma, avg_mse
                
    return best_gamma, best_sigma, mse_results

# make predictions using the kernel matrix and alpha values
def predict(kernel_matrix, alpha_values):
    return np.dot(kernel_matrix, alpha_values)

# evaluate kernel ridge regression on training and testing data
def evaluate(training_data, testing_data, gamma, sigma):
    # separate features and targets for training and testing data
    train_features = training_data[:, :-1]
    train_targets = training_data[:, -1]
    test_features = testing_data[:, :-1]
    test_targets = testing_data[:, -1]

    # compute the kernel matrix for training data and calculate alpha values
    training_kernel_matrix = compute_kernel_matrix(train_features, sigma)
    alpha_values = kernel_ridge_regression(training_kernel_matrix, train_targets, gamma)

    # compute the kernel matrix for testing data
    testing_kernel_matrix = compute_test_kernel_matrix(train_features, test_features, sigma)
    predictions = predict(testing_kernel_matrix, alpha_values)

    # calculate mean squared errors for training and testing data
    training_predictions = predict(training_kernel_matrix, alpha_values)
    training_mse = np.mean((training_predictions - train_targets) ** 2)
    testing_mse = np.mean((predictions - test_targets) ** 2)
    return training_mse, testing_mse

# plot cross-validation results as a function of gamma and sigma
def plot_cross_validation_results(mse_results):
    gamma_values, sigma_values, mse_values = zip(*mse_results)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(np.log2(sigma_values), np.log2(gamma_values), mse_values, c=mse_values, cmap='viridis')
    ax.set_xlabel('log2(sigma)')  # x-axis represents sigma
    ax.set_ylabel('log2(gamma)')  # y-axis represents gamma
    ax.set_zlabel('mse')  # z-axis represents mean squared error
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    plt.show()

# perform random splits and summarize results for all methods
def repeat_random_splits(data, gamma_values, sigma_values, num_splits=20):
    results = []

    # baseline regression
    train_mse_list, test_mse_list = [], []
    for _ in range(num_splits):
        train_set, test_set = split_data_set(data)
        train_mse, test_mse = baseline_regression(train_set, test_set)
        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)
    results.append((
        "Naive Regression",
        np.mean(train_mse_list),
        np.std(train_mse_list),
        np.mean(test_mse_list),
        np.std(test_mse_list),
    ))

    # linear regression (single attributes)
    # iterate over all attributes
    for k in range(data.shape[1] - 1):
        train_mse_list, test_mse_list = [], []
        for _ in range(num_splits):
            train_set, test_set = split_data_set(data)
            train_mse, test_mse = linear_regression_single_attribute(train_set, test_set, k)
            train_mse_list.append(train_mse)
            test_mse_list.append(test_mse)
        results.append((
            f"Linear Regression (attribute {k+1})",
            np.mean(train_mse_list),
            np.std(train_mse_list),
            np.mean(test_mse_list),
            np.std(test_mse_list),
        ))

    # linear regression (all attributes)
    train_mse_list, test_mse_list = [], []
    for _ in range(num_splits):
        train_set, test_set = split_data_set(data)
        train_mse, test_mse = linear_regression_all_attributes(train_set, test_set)
        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)
    results.append((
        "Linear Regression (all attributes)",
        np.mean(train_mse_list),
        np.std(train_mse_list),
        np.mean(test_mse_list),
        np.std(test_mse_list),
    ))

    # kernel ridge regression
    train_mse_list, test_mse_list = [], []
    for _ in range(num_splits):
        train_set, test_set = split_data_set(data)
        best_gamma, best_sigma, _ = five_fold_cross_validation(train_set, gamma_values, sigma_values)
        train_mse, test_mse = evaluate(train_set, test_set, best_gamma, best_sigma)
        train_mse_list.append(train_mse)
        test_mse_list.append(test_mse)
    results.append((
        "Kernel Ridge Regression",
        np.mean(train_mse_list),
        np.std(train_mse_list),
        np.mean(test_mse_list),
        np.std(test_mse_list),
    ))

    return results

# generate the summary table
def generate_summary_table(data, gamma_values, sigma_values, num_splits=20):
    summary_results = repeat_random_splits(data, gamma_values, sigma_values, num_splits)
    summary_df = pd.DataFrame(summary_results, columns=["Method", "MSE Train ± σ", "MSE Test ± σ"])
    summary_df["MSE Train ± σ"] = summary_df.apply(lambda row: f"{row[1]:.4f} ± {row[2]:.4f}", axis=1)
    summary_df["MSE Test ± σ"] = summary_df.apply(lambda row: f"{row[3]:.4f} ± {row[4]:.4f}", axis=1)
    return summary_df

# main script
if __name__ == "__main__":
    # load data
    data = load_data()

    # split into training (2/3) and testing (1/3)
    training_data, testing_data = split_data_set(data)

    # define parameter ranges for gamma and sigma
    gamma_values = [2 ** i for i in range(-40, -25)]
    sigma_values = [2 ** i for i in range(7, 14)]

    # part 5(a): find best gamma and sigma using cross-validation
    best_gamma, best_sigma, mse_results = five_fold_cross_validation(training_data, gamma_values, sigma_values)
    print(f"best gamma: {best_gamma}, best sigma: {best_sigma}")

    # part 5(b): plot cross-validation errors
    plot_cross_validation_results(mse_results)

    # part 5(c): evaluate on training and testing sets
    training_mse, testing_mse = evaluate(training_data, testing_data, best_gamma, best_sigma)
    print(f"training mse: {training_mse}, testing mse: {testing_mse}")

    # part 5(d): generate and display the summary table
    final_summary_table = generate_summary_table(data, gamma_values, sigma_values)
    print(final_summary_table)
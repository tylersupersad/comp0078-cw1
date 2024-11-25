import numpy as np
import matplotlib.pyplot as plt
from question1 import compute_regression_weights
from question2 import generate_dataset

# maps data to sine basis features for all degrees up to max_degree
def sine_basis_feature_map(data_x, max_degree):
    # ensure input is a 2D array
    data_x = np.array(data_x).reshape(-1, 1)

    # generate sine basis features for degrees 1 to max_degree
    degrees = np.arange(1, max_degree + 1)
    sine_bases = np.sin(np.pi * degrees * data_x)

    # return a list where each element contains sine basis features up to degree k
    return [sine_bases[:, :k] for k in degrees]

# computes MSE using precomputed sine basis features
def compute_mse_sine_basis(train_features, y_train, test_features, y_test):
    # compute regression weights using training data
    weights = compute_regression_weights(train_features, y_train)

    # make predictions on test data
    predictions = test_features.dot(weights)

    # calculate and return the MSE
    mse = np.mean((predictions - y_test) ** 2)
    return mse

# computes the ln(MSE) for all degrees from 1 to max_degree
def compute_ln_mse_over_degrees(x_train, y_train, x_test, y_test, max_degree):
    # precompute sine basis features for training and test datasets
    train_features_list = sine_basis_feature_map(x_train, max_degree)
    test_features_list = sine_basis_feature_map(x_test, max_degree)
    
    # list to store ln(MSE) for each degree
    ln_mse = []  

    # loop through each degree and compute ln(MSE)
    for train_features, test_features in zip(train_features_list, test_features_list):
        mse = compute_mse_sine_basis(train_features, y_train, test_features, y_test)
        
        # store the natural logarithm of MSE
        ln_mse.append(np.log(mse))  

    return ln_mse

def plot_ln_mse(ln_mse, title, label):
    # generate degree values corresponding to ln(MSE)
    degrees = np.arange(1, len(ln_mse) + 1)

    # create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, ln_mse, marker='o', linestyle='-', label=label)

    # add labels, title, and legend
    plt.xlabel("Basis Dimension", fontsize=12)
    plt.ylabel("ln(MSE)", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True) 
    plt.xticks(degrees)
    plt.show()

# computes the average ln(MSE) for training and test data over multiple runs
def average_ln_mse(num_runs, dataset_size, test_size, max_degree):
    # initialize cumulative MSE storage for training and test data
    cumulative_train_mse = np.zeros(max_degree)
    cumulative_test_mse = np.zeros(max_degree)

    # repeat the experiment num_runs times
    for _ in range(num_runs):
        # generate training and test datasets
        x_train, y_train = generate_dataset(dataset_size)
        x_test, y_test = generate_dataset(test_size)

        # precompute sine basis features for all degrees
        train_features_list = sine_basis_feature_map(x_train, max_degree)
        test_features_list = sine_basis_feature_map(x_test, max_degree)
        
        # compute MSE for each degree and accumulate results
        for idx, (train_features, test_features) in enumerate(zip(train_features_list, test_features_list)):
            train_mse = compute_mse_sine_basis(train_features, y_train, train_features, y_train)
            cumulative_train_mse[idx] += train_mse
            test_mse = compute_mse_sine_basis(train_features, y_train, test_features, y_test)
            cumulative_test_mse[idx] += test_mse

    # compute average ln(MSE) for training and test data
    avg_train_ln_mse = np.log(cumulative_train_mse / num_runs)
    avg_test_ln_mse = np.log(cumulative_test_mse / num_runs)

    return avg_train_ln_mse, avg_test_ln_mse

# plots the average ln(MSE) for training and test data over multiple runs
def plot_average_ln_mse(avg_train_ln_mse, avg_test_ln_mse, max_degree):
    # generate degree values
    degrees = np.arange(1, max_degree + 1)

    # create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(degrees, avg_train_ln_mse, marker='o', linestyle='-', label="Average Training Error")
    plt.plot(degrees, avg_test_ln_mse, marker='o', linestyle='-', label="Average Test Error")

    # add labels, title, and legend
    plt.xlabel("Basis Dimension", fontsize=12)
    plt.ylabel("ln(Average MSE)", fontsize=12)
    plt.title("Average ln(MSE) over Multiple Runs", fontsize=14)
    plt.legend()
    plt.grid(True) 
    plt.xticks(degrees)
    plt.show()

if __name__ == "__main__":
    # number of samples in the training dataset
    dataset_size = 30    
    # number of samples in the test dataset     
    test_size = 1000
    # maximum degree for the sine basis          
    max_degree = 18
    # number of runs for averaging           
    num_runs = 100          

    # part (b): compute and plot ln(MSE) for training data
    x_train, y_train = generate_dataset(dataset_size)
    train_ln_mse = compute_ln_mse_over_degrees(x_train, y_train, x_train, y_train, max_degree)
    plot_ln_mse(train_ln_mse, "Training Error (Sine Basis)", label="Training Error")

    # part (c): compute and plot ln(MSE) for test data
    x_test, y_test = generate_dataset(test_size)
    test_ln_mse = compute_ln_mse_over_degrees(x_train, y_train, x_test, y_test, max_degree)
    plot_ln_mse(test_ln_mse, "Test Error (Sine Basis)", label="Test Error")

    # part (d): compute and plot average ln(MSE) over multiple runs
    avg_train_ln_mse, avg_test_ln_mse = average_ln_mse(num_runs, dataset_size, test_size, max_degree)
    plot_average_ln_mse(avg_train_ln_mse, avg_test_ln_mse, max_degree)
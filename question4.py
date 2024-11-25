import csv
import numpy as np
import matplotlib.pyplot as plt
from question1 import compute_regression_weights

# splits the dataset into training (2/3) and testing sets (1/3)
def split_data_set(data, test_fraction=1/3):
    n = len(data)
    test_size = int(n * test_fraction)
    
    # randomly select test indices
    indices = np.random.choice(n, size=test_size, replace=False)
    test_set = data[indices]
    
    # remove test indices to create the train set
    train_set = np.delete(data, indices, axis=0)
    return train_set, test_set

# computes MSE for given weights and data
def calculate_mse(weights, x, y):
    predictions = x.dot(weights)
    return np.mean((predictions - y) ** 2)

# adds a column of ones to the dataset
def add_bias_term(x):
    return np.hstack((x, np.ones((x.shape[0], 1))))

# computes baseline regression using the mean of training targets
def baseline_regression(training_set, testing_set):
    # target values from training set
    training_y = training_set[:, -1]
    # target values from testing set
    testing_y = testing_set[:, -1]
    
    # mean of training targets
    y_mean = np.mean(training_y)

    # compute MSE for baseline predictor
    training_mse = np.mean((training_y - y_mean) ** 2)
    testing_mse = np.mean((testing_y - y_mean) ** 2)

    return training_mse, testing_mse

# performs linear regression using only the k-th attribute and logs results
def linear_regression_single_attribute(training_set, testing_set, k):
    # prepare training data
    x_train = add_bias_term(training_set[:, [k]])
    y_train = training_set[:, -1]

    # compute regression weights
    weights = compute_regression_weights(x_train, y_train)
    # compute training MSE
    training_mse = calculate_mse(weights, x_train, y_train)

    # prepare testing data
    x_test = add_bias_term(testing_set[:, [k]])
    y_test = testing_set[:, -1]

    # compute testing MSE
    testing_mse = calculate_mse(weights, x_test, y_test)

    return training_mse, testing_mse

# logs and displays MSE for EACH attribute for training and testing
def calculate_mse_per_attribute(data):
    # exclude target variable
    num_features = data.shape[1] - 1

    # split data into training and testing sets
    train_set, test_set = split_data_set(data)

    print("\nMSE for Each Attribute:")
    print(f"{'Attribute':<10}{'Training MSE':<20}{'Testing MSE'}")
    print("-" * 40)

    # iterate over each attribute
    for k in range(num_features):
        # compute MSE for current attribute
        train_mse, test_mse = linear_regression_single_attribute(train_set, test_set, k)
        # display results
        print(f"{k:<10}{train_mse:<20.4f}{test_mse:.4f}")

# performs linear regression using all attributes in the dataset
def linear_regression_all_attributes(training_set, testing_set):
    # prepare training data
    x_train = add_bias_term(training_set[:, :-1])
    y_train = training_set[:, -1]

    # compute regression weights
    weights = compute_regression_weights(x_train, y_train)
    # compute training MSE
    training_mse = calculate_mse(weights, x_train, y_train)

    # prepare testing data
    x_test = add_bias_term(testing_set[:, :-1])
    y_test = testing_set[:, -1]

    # compute testing MSE
    testing_mse = calculate_mse(weights, x_test, y_test)

    return training_mse, testing_mse

# calculates average testing MSE for each attribute over multiple runs
def average_testing_mse_per_attribute(data, num_runs=20):
    # exclude target variable
    num_features = data.shape[1] - 1
    avg_mse = []

    for k in range(num_features):
        mse_sum = 0
        # repeat for specified number of runs
        for _ in range(num_runs):
            train_set, test_set = split_data_set(data)
            _, test_mse = linear_regression_single_attribute(train_set, test_set, k)
            mse_sum += test_mse
            
        # average the MSE for the k-th feature
        avg_mse.append(mse_sum / num_runs)

    # plot average testing MSE for each attribute
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_features), avg_mse, marker='o', linestyle='-', label="Average MSE")
    plt.xlabel("Attribute Index")
    plt.ylabel("Average Testing MSE")
    plt.title("Average Testing MSE for Each Attribute")
    plt.grid(True)
    plt.legend()
    plt.show()

# computes average MSE for training and testing data over multiple runs
def average_mse_all_attributes(data, num_runs=20):
    train_mse_sum = 0
    test_mse_sum = 0

    # repeat for specified number of runs
    for _ in range(num_runs):
        train_set, test_set = split_data_set(data)
        train_mse, test_mse = linear_regression_all_attributes(train_set, test_set)
        train_mse_sum += train_mse
        test_mse_sum += test_mse

    # compute average MSE
    avg_train_mse = train_mse_sum / num_runs
    avg_test_mse = test_mse_sum / num_runs

    return avg_train_mse, avg_test_mse

# load data from boston.csv
def load_data(data_file="boston.csv"):
    data = []
    try:
        # open and read csv file
        with open(data_file, 'r') as file:
            reader = csv.reader(file)
            
            # skip header
            next(reader)

            # append rows to the data list
            for row in reader:
                data.append([float(x) for x in row])

        # convert list to a numpy array
        return np.array(data)
    
    # exception error
    except FileNotFoundError:
        print(f"Error: File '{data_file}' not found.")
        exit(1)

if __name__ == "__main__":
    # load dataset
    data = load_data()
    
    # display dataset shape
    print("Data shape:", data.shape)

    # part (a): baseline regression
    train_set, test_set = split_data_set(data)
    baseline_train_mse, baseline_test_mse = baseline_regression(train_set, test_set)
    print(f"Baseline Regression MSE (Training, Testing): {baseline_train_mse:.4f}, {baseline_test_mse:.4f}")

    # part (c): calculate and display MSE for each attribute
    calculate_mse_per_attribute(data)
    average_testing_mse_per_attribute(data)

    # part (d): average testing MSE with all attributes
    avg_train_mse, avg_test_mse = average_mse_all_attributes(data)
    print(f"Average MSE using all attributes (Training, Testing): {avg_train_mse:.4f}, {avg_test_mse:.4f}")
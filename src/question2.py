import numpy as np
import matplotlib.pyplot as plt
from question1 import create_design_matrix, compute_regression_weights

# part (a): generate noisy dataset
def generate_dataset(num_samples, sigma=0.07):
    # generate x sampled uniformly from [0, 1] and y = sin^2(2πx) + noise
    x_values = np.random.uniform(0, 1, num_samples)
    y_values = np.sin(2 * np.pi * x_values) ** 2 + np.random.normal(0, sigma, num_samples)
    return x_values, y_values

# part (ii): fit a polynomial of a given degree and predict y-values
def fit_polynomial_and_predict(x_train, y_train, x_predict, degree):
    # build design matrices for training and prediction
    design_matrix_train = create_design_matrix(x_train, degree)
    design_matrix_predict = create_design_matrix(x_predict, degree)
    
    # compute regression weights
    weights = compute_regression_weights(design_matrix_train, y_train)
    
    return design_matrix_predict.dot(weights)

# compute MSE for a specific polynomial degree
def compute_mse(x_train, y_train, x_predict, y_predict, degree):
    # Predict y-values using the fitted polynomial
    y_pred = fit_polynomial_and_predict(x_train, y_train, x_predict, degree)
    # Compute Mean Squared Error
    return np.mean((y_predict - y_pred) ** 2)

# part (b) and (c): compute ln(MSE) for training or test data
def compute_ln_mse(x_data, y_data, x_predict, y_predict, max_degree):
    # calculate natural log of MSE for polynomial fits up to max_degree
    errors = []
    for degree in range(1, max_degree + 1):
        mse = compute_mse(x_data, y_data, x_predict, y_predict, degree)
        errors.append(np.log(mse))
    return errors

# part (b) and (c): plot ln(MSE) for training or test data
def plot_error(x_train, y_train, x_test, y_test, title, color="blue"):
    # compute and plot ln(MSE) as a function of polynomial degree
    errors = compute_ln_mse(x_train, y_train, x_test, y_test, max_degree=18)
    plt.plot(range(1, 19), errors, marker='o', label=title, color=color)
    plt.xlabel("Polynomial Degree")
    plt.ylabel("ln(MSE)")
    plt.title(f"{title} vs Polynomial Degree")
    plt.legend()
    plt.show()

# part (a): plot sin^2(2πx) function with noisy data points
def plot_function_and_data():
    # generate smooth x-values for the base function
    x_smooth = np.linspace(0, 1, 500)
    y_smooth = np.sin(2 * np.pi * x_smooth) ** 2
    # generate noisy data
    x_data, y_data = generate_dataset(30)
    
    # plot base function and noisy data points
    plt.plot(x_smooth, y_smooth, label=r"$\sin^2(2\pi x)$", color="blue")
    plt.scatter(x_data, y_data, color="red", label="Noisy Data", alpha=0.6)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(r"Function $\sin^2(2\pi x)$ with Noisy Data")
    plt.show()

# part (ii): fit polynomial curves and plot them
def fit_and_plot_polynomials(x_data, y_data, degrees=[2, 5, 10, 14, 18]):
    # generate smooth x-values for plotting polynomial fits
    x_smooth = np.linspace(0, 1, 500)
    y_smooth = np.sin(2 * np.pi * x_smooth) ** 2
    # plot base function
    plt.plot(x_smooth, y_smooth, label=r"$\sin^2(2\pi x)$", color="blue")
    # plot noisy data points
    plt.scatter(x_data, y_data, color="red", label="Noisy Data", alpha=0.6)

    # fit and plot polynomials for specified degrees
    for degree in degrees:
        y_plot = fit_polynomial_and_predict(x_data, y_data, x_smooth, degree)
        plt.plot(x_smooth, y_plot, label=f"Degree {degree}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Polynomial Fits of Different Degrees with Noisy Data")
    plt.show()

# part (d): compute average ln(MSE) for training and test data over multiple runs
def average_ln_mse_over_runs(num_runs=100, max_degree=18):
    # store cumulative training errors
    cumulative_training_errors = np.zeros(max_degree) 
    # store cumulative test errors
    cumulative_test_errors = np.zeros(max_degree) 

    for _ in range(num_runs):
        # generate training dataset of 30 points
        x_train, y_train = generate_dataset(30) 
        # generate test dataset of 1000 points
        x_test, y_test = generate_dataset(1000) 

        for degree in range(1, max_degree + 1):
            # compute training and test MSE for each degree
            train_mse = compute_mse(x_train, y_train, x_train, y_train, degree)
            test_mse = compute_mse(x_train, y_train, x_test, y_test, degree)
            cumulative_training_errors[degree - 1] += train_mse
            cumulative_test_errors[degree - 1] += test_mse

    # compute log of average training and test errors
    log_avg_training_errors = np.log(cumulative_training_errors / num_runs)
    log_avg_test_errors = np.log(cumulative_test_errors / num_runs)
    
    return log_avg_training_errors, log_avg_test_errors

# part (d): plot average ln(MSE) for training and test data
def plot_smoothed_errors():
    # get smoothed ln(MSE) for training and test data
    log_avg_training_errors, log_avg_test_errors = average_ln_mse_over_runs()
    # polynomial degrees
    degrees = np.arange(1, len(log_avg_training_errors) + 1) 
    
    # plot smoothed training error
    plt.plot(degrees, log_avg_training_errors, marker='o', label="Training Error")
    # plot smoothed test error
    plt.plot(degrees, log_avg_test_errors, marker='o', label="Test Error", color="orange")
    
    plt.xlabel("Polynomial Degree")
    plt.ylabel("ln(Average MSE)")
    plt.title("Smoothed Training and Test Errors vs Polynomial Degree")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # part (ai): plot function and noisy data
    plot_function_and_data()
    
    # generate training dataset
    x_data, y_data = generate_dataset(30) 
    
    # part (aii): fit and plot polynomials
    fit_and_plot_polynomials(x_data, y_data) 
    
    # part (b): plot training error
    plot_error(x_data, y_data, x_data, y_data, title="Training Error") 
    
    # generate test dataset
    x_test, y_test = generate_dataset(1000) 
    
    # part (c): plot test error
    plot_error(x_data, y_data, x_test, y_test, title="Test Error", color="orange") 
    
    # part (d): Plot smoothed training and test errors
    plot_smoothed_errors()  
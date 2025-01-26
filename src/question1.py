import numpy as np
import math
import matplotlib.pyplot as plt

# create a design matrix for polynomial regression
def create_design_matrix(data_x, k):
    # initialize design matrix
    result = np.zeros((len(data_x), k))
    for row, x in enumerate(data_x):
        for column in range(k):
            # compute x raised to the power of column index
            result[row][column] = math.pow(x, column)
    return result

# compute regression weights using the formula: (X^T * X)^-1 * X^T * y
def compute_regression_weights(mapped_data, data_y):
    # transpose the design matrix
    transpose = mapped_data.T
    # compute X^T * X
    temp = transpose.dot(mapped_data)
    # calculate pseudoinverse for numerical stability
    pseudo_inverse = np.linalg.pinv(temp) 
    
    # compute weights
    result = pseudo_inverse.dot(transpose).dot(data_y)
    # return rounded coefficients for better readability
    return np.around(result, 2)  

# 1(a): plot polynomial fits for different degrees
def plot_polynomial_fits(arr_x, arr_y):
    # create a figure for the plots
    plt.figure(figsize=(8, 4))
    
    # loop through polynomial degrees from 1 to 4
    for k in range(1, 5):
        # create design matrix for degree k
        mapped_data = create_design_matrix(arr_x, k)
        # compute regression weights for degree k
        weights = compute_regression_weights(mapped_data, arr_y)
        
        # generate x values for a smooth curve
        x = np.linspace(0, 5, 100)
        # calculate y values using the regression weights
        y = sum(c * x**i for i, c in enumerate(weights))
        
        # plot the curve for the current degree
        plt.plot(x, y, linewidth=1.5, label=f'k={k}')
    
    # add labels, legend, and title to the plot
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Polynomial Fits of Different Degrees')
    # display the plot
    plt.show()

# 1(b): print polynomial equations for different degrees
def print_polynomial_equations(arr_x, arr_y):
    # loop through polynomial degrees from 1 to 4
    for k in range(1, 5):
        # create design matrix for degree k
        mapped_data = create_design_matrix(arr_x, k)
        # compute regression weights for degree k
        weights = compute_regression_weights(mapped_data, arr_y)
        
        # construct the polynomial equation as a string
        equation_terms = []
        for i in range(k):
            if i == 0:
                equation_terms.append(f"{weights[i]:.2f}")
            else:
                equation_terms.append(f"{weights[i]:.2f} * x^{i}")
        equation = " + ".join(equation_terms)
        # print the polynomial equation
        print(f"degree {k} polynomial: y = {equation}")

# 1(c): calculate and print mean squared error (mse) for different degrees
def calculate_mse(arr_x, arr_y):
    # loop through polynomial degrees from 1 to 4
    for k in range(1, 5):
        # create design matrix for degree k
        mapped_data = create_design_matrix(arr_x, k)
        # compute regression weights for degree k
        weights = compute_regression_weights(mapped_data, arr_y)
        # calculate mse for the current polynomial fit
        error = ((mapped_data.dot(weights) - arr_y).T.dot(mapped_data.dot(weights) - arr_y)) / len(arr_y)
        # print the mse for the current degree
        print(f"mse for k={k}: {error:.4f}")

if __name__ == '__main__':
    # input data points
    data_x = [1, 2, 3, 4]
    # corresponding target values 
    data_y = [3, 2, 0, 5]  
    
    # convert x and y data to numpy arrays
    arr_x = np.array(data_x)  
    arr_y = np.array(data_y)  
    
    # 1(a) generate plot
    plot_polynomial_fits(arr_x, arr_y)
    
    # 1(b) print polynomial equations
    print_polynomial_equations(arr_x, arr_y)
    
    # 1(c) calculate and print mse
    calculate_mse(arr_x, arr_y)
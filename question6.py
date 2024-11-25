import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# generate random data points and assign random binary labels
def generate_random_data(num_samples=100):
    # create random points in [0,1]^2
    data_points = np.random.rand(num_samples, 2)
    # assign random binary labels (0 or 1) to each point
    data_labels = np.random.choice([0, 1], size=num_samples)
    return data_points, data_labels

# compute the label for a new point using knn
def predict_label_knn(target_point, data_points, data_labels, num_neighbors):
    # calculate euclidean distances from the target point to all data points
    distances = distance.cdist([target_point], data_points, metric='euclidean')[0]
    # identify the indices of the k nearest neighbors
    nearest_indices = np.argsort(distances)[:num_neighbors]
    # retrieve the labels of the k nearest neighbors
    nearest_labels = data_labels[nearest_indices]
    # determine the majority label among the k neighbors
    majority_label = np.argmax(np.bincount(nearest_labels))
    return majority_label

# generate the decision boundary for the hypothesis
def compute_decision_boundary(data_points, data_labels, resolution=100, num_neighbors=3):
    # create a grid of x and y values over [0,1]^2
    x_values = np.linspace(0, 1, resolution)
    y_values = np.linspace(0, 1, resolution)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    
    # combine x and y into 2D points
    grid_points = np.c_[x_grid.ravel(), y_grid.ravel()] 

    # predict labels for all grid points using k-NN
    grid_labels = np.array([predict_label_knn(point, data_points, data_labels, num_neighbors) for point in grid_points])
    return x_grid, y_grid, grid_labels.reshape(x_grid.shape)

# visualize the hypothesis as a decision boundary
def plot_decision_boundary():
    # generate 100 random samples of training data
    data_points, data_labels = generate_random_data()

    # generate the decision boundary for the hypothesis
    x_grid, y_grid, decision_boundary = compute_decision_boundary(data_points, data_labels)

    # plot the boundary and training points
    plt.figure(figsize=(8, 6))
    plt.contourf(x_grid, y_grid, decision_boundary, levels=[-0.5, 0.5, 1.5], colors=['white', 'turquoise'], alpha=0.6)
    plt.scatter(data_points[np.where(data_labels == 0), 0], data_points[np.where(data_labels == 0), 1], color='blue', label='Label 0')
    plt.scatter(data_points[np.where(data_labels == 1), 0], data_points[np.where(data_labels == 1), 1], color='green', label='Label 1')
    plt.title("Visualization of $h_{S,v}$ with $|S|=100$ and $v=3$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # produce a visualization of an hS,v similar to figure 1
    plot_decision_boundary()
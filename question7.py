import numpy as np
import matplotlib.pyplot as plt
from question6 import generate_random_data, predict_label_knn 

# generate noisy data based on h_S,3
def generate_noisy_data(num_points, h_s3, p_heads=0.8):
    # generate random points in [0, 1]^2
    points = np.random.rand(num_points, 2)
    labels = []

    for point in points:
        # use h_s3 for the label with probability p_heads
        if np.random.rand() < p_heads:
            labels.append(h_s3(point))
        else:
            # assign a random label with probability 1 - p_heads
            labels.append(np.random.choice([0, 1]))
    
    return points, np.array(labels)

# define the hypothesis h_S,3
def generate_h_s3(points, labels):
    def h_s3(point):
        # predict label using k=3
        return predict_label_knn(point, points, labels, num_neighbors=3)  # reuse predict_label_knn
    return h_s3

# compute generalization error for different k values
def compute_generalization_error(k_values, runs, train_size, test_size):
    errors = []

    # protocol a: iterate over different k values (1 to 49)
    for k in k_values:
        error_runs = []

        # protocol a: perform 100 runs for each k
        for _ in range(runs):
            # protocol a: sample h from ph by generating random points and labels
            points, labels = generate_random_data(100)  
            h_s3 = generate_h_s3(points, labels)  # generate the hypothesis h_S,3

            # protocol a: build a k-NN model with 4000 training points
            train_points, train_labels = generate_noisy_data(train_size, h_s3)

            # protocol a: run k-NN on 1000 test points to estimate generalization error
            test_points, test_labels = generate_noisy_data(test_size, h_s3)

            # compute the error on the test set
            test_errors = 0
            for i in range(test_size):
                # protocol a: predict test label using k-NN
                predicted_label = predict_label_knn(test_points[i], train_points, train_labels, num_neighbors=k)
                if predicted_label != test_labels[i]:
                    test_errors += 1

            # store error rate for this run
            error_runs.append(test_errors / test_size)

        # protocol a: compute the mean generalization error over 100 runs
        errors.append(np.mean(error_runs))

    return errors

# visualization of generalization error vs. k
def visualize_generalization_error():
    # k values from 1 to 49
    k_values = list(range(1, 50))
    # 30 runs for averaging
    runs = 30 
    # number of training points
    train_size = 4000
    # number of testing points
    test_size = 1000

    # compute generalization errors
    errors = compute_generalization_error(k_values, runs, train_size, test_size)

    # plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, errors, marker='o', linestyle='-', color='b', label='Generalization Error')
    plt.title("Estimated Generalization Error of k-NN as a Function of k")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Generalization Error")
    plt.xticks(range(1, 50, 2))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # visualize the generalization error as a function of k
    visualize_generalization_error()
import numpy as np
import matplotlib.pyplot as plt
from question6 import generate_random_data, predict_label_knn 
from question7 import generate_noisy_data, generate_h_s3  

# compute optimal k for given training sizes
def compute_optimal_k(m_values, k_values, runs, test_size):
    # store average optimal k for each training size
    optimal_ks = [] 

    # protocol b: iterate over training sizes
    for m in m_values:
        # store optimal k for each run
        k_for_runs = [] 
        
        # protocol b: perform 100 runs for each training size
        for _ in range(runs): 
            # protocol b: sample h_S,3 (hypothesis) for this run
            points, labels = generate_random_data(100)
            h_s3 = generate_h_s3(points, labels)

            # protocol b: generate m training points and test set
            train_x, train_y = generate_noisy_data(m, h_s3)
            test_x, test_y = generate_noisy_data(test_size, h_s3)

            # evaluate errors for all k
            errors = []
            # protocol b: iterate over k values
            for k in k_values: 
                test_errors = 0
                
                for i in range(test_size):
                    # protocol b: compute generalization error for k-NN
                    predicted_label = predict_label_knn(test_x[i], train_x, train_y, k)
                    
                    if predicted_label != test_y[i]:
                        test_errors += 1

                errors.append(test_errors / test_size)

            # protocol b: find k with minimal generalization error for this run
            optimal_k = k_values[np.argmin(errors)]
            k_for_runs.append(optimal_k)

        # protocol b: average the optimal k over all 100 runs
        optimal_ks.append(np.mean(k_for_runs))

    return optimal_ks

# visualize optimal k
def visualize_optimal_k():
    # training sizes
    m_values = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000] 
    # k values to test
    k_values = list(range(1, 50)) 
    # number of runs
    runs = 100  
    # number of test points
    test_size = 1000  

    # compute optimal ks
    optimal_ks = compute_optimal_k(m_values, k_values, runs, test_size)

    # plot results
    plt.figure(figsize=(8, 6))
    plt.plot(m_values, optimal_ks, marker='o', linestyle='-', color='b', label='Optimal k')
    plt.title("Optimal k as a Function of Training Size")
    plt.xlabel("Training Size (m)")
    plt.ylabel("Optimal k")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # produce the visualization for optimal k
    visualize_optimal_k()
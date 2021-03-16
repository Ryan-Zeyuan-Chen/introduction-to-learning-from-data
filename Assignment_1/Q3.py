import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from time import time
import sys
sys.path.append('../')
from data.data_utils import load_dataset

def l2_dist(x_train, x_test):
    return np.sqrt(np.sum(np.square(x_train-x_test), axis=1))

def knn_regression_bf(x_train, y_train, x_test, k):
    dist = l2_dist(x_train, x_test)
    i_nn = np.argpartition(dist, kth=k)[:k]
    y_test = np.average(y_train[i_nn])
    return y_test

def knn_regression_kdt(x_train, y_train, x_test, k):
    kdt = KDTree(x_train, metric='euclidean')
    i_nn = kdt.query(x_test, k=k, return_distance=False)
    y_test = np.average(y_train[i_nn], axis=1)
    return y_test

def graph_time(d, elapsed_bf, elapsed_kdt):
    plt.figure()
    plt.plot(d, elapsed_bf, d, elapsed_kdt)
    plt.xlabel('Dimension')
    plt.ylabel('k-NN Regression Run-Time (s)')
    plt.title('k-NN Regression Algorithm Run-Time Performance')
    plt.legend(['Brute-Force', 'K-D Tree'])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    print('k-NN Regression Algorithm using the K-D Tree Data Structure\n\nPerforming prediction on rosenbrock test set using chosen k value of 5:')
    elapsed_bf = []
    elapsed_kdt = []
    d = range(2,33)
    for i in d:
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=5000, d=i)
        y_predict_bf = []
        print('Running the brute-force approach for d =', i, '...')
        start = time()
        for j in range(len(x_test)):
            y_predict_bf.append(knn_regression_bf(x_train, y_train, x_test[j], 5))
        finish = time()
        elapsed_bf.append(finish - start)
        print('Running the K-D tree algorithm for d =', i, '...')
        start = time()
        y_predict_kdt = knn_regression_kdt(x_train, y_train, x_test, 5)
        finish = time()
        elapsed_kdt.append(finish - start)
    graph_time(d, elapsed_bf, elapsed_kdt)

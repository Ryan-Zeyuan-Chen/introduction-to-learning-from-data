import numpy as np
import sys
sys.path.append('../')
from data.data_utils import load_dataset

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
# x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')

def l1_dist(x_train, x_test):
    return np.sum(np.abs(x_train-x_test), axis=1)

def l2_dist(x_train, x_test):
    return np.sqrt(np.sum(np.square(x_train-x_test), axis=1))

def knn_classifier(x_train, y_train, x_test, k):
    dist = l2_dist(x_train, x_test)
    i_nn = np.argpartition(dist, kth=k)[:k]
    vote, count = np.unique(y_train[i_nn], return_counts=True, axis=0)
    y_test = vote[np.argmax(count)]
    return y_test

def knn_validate(x_train, y_train, x_valid, y_valid, k):
    y_predict = []
    correct = 0
    for j in range(len(x_valid)):
        y_predict.append(knn_classifier(x_train, y_train, x_valid[j], k))
        if np.array_equal(y_predict[j], y_valid[j]):
            correct += 1
    accuracy = correct/len(x_valid)*100
    return accuracy

if __name__ == "__main__":
    print('k-NN Classification Algorithm\n\nPerforming validation for k = 1 to k = 30:')
    v_accuracy = []
    for i in range(30):
        v_accuracy.append((i+1, knn_validate(x_train, y_train, x_valid, y_valid, i+1)))
        print('k =', v_accuracy[i][0], "\tAccuracy =", v_accuracy[i][1], "%")
    v_accuracy.sort(reverse = True, key=lambda x: x[1])
    k = v_accuracy[0][0]
    print('\nThe estimated k value is', k, 'and the maximum validation accuracy is', v_accuracy[0][1], "%\n")
    print('Performing prediction on test set using chosen k value of', k, ':')
    t_accuracy = knn_validate(x_train, y_train, x_test, y_test, k)
    print('The test accuracy for k =', k, 'is', t_accuracy, '%')

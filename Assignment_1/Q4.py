import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('../')
from data.data_utils import load_dataset

datasets = ['mauna_loa', 'rosenbrock',  'pumadyn32nm', 'iris', 'mnist_small']

def accuracy(y_test, y_predict):
    correct = 0
    for i in range(len(y_test)):
        if np.array_equal(y_test[i], y_predict[i]):
            correct += 1
    accuracy = correct/len(y_test)*100
    return accuracy

def linear_regression(x_train, y_train, x_test, type):
    u_1, s_1, v_t = np.linalg.svd(x_train, full_matrices=False)
    singular_1 = np.diag(s_1)
    w = np.dot(v_t.T, np.dot(np.linalg.inv(singular_1), np.dot(u_1.T, y_train)))
    y_test = np.dot(x_test, w)
    if type:
        for i in range(len(y_test)):
            index = np.argmax(y_test[i,:])
            y_test[i,:] = False
            y_test[i,index] = True
    return y_test

def graph_test(x_test, y_test, y_predict):
    plt.figure()
    plt.plot(x_test, y_test, x_test, y_predict)
    plt.xlabel('x_test')
    plt.ylabel('y_test')
    plt.title('Linear Regression Algorithm Mauna Loa Test Set Prediction')
    plt.legend(['Ground Truth', 'Prediction'])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    print('Linear Regression Algorithm\n\nPerforming prediction on test set by minimizing the least-squares loss function using SVD:')
    for data in datasets:
        if data == 'mauna_loa' or data == 'rosenbrock' or data == 'pumadyn32nm':
            if data == 'rosenbrock':
                x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(data, n_train=5000, d=2)
            else:
                x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(data)
            type = 0 #type = 0 for regression datasets
        else:
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(data)
            type = 1 #type = 1 for classification datasets
        x_train = np.vstack([x_valid, x_train])
        y_train = np.vstack([y_valid, y_train])
        if not type:
            y_predict = linear_regression(x_train, y_train, x_test, type)
            rmse = mean_squared_error(y_test, y_predict, squared=False)
            print('The test RMSE value for the', data, 'dataset is', rmse)
            if data == 'mauna_loa':
                graph_test(x_test, y_test, y_predict)
        else:
            y_predict = linear_regression(x_train, y_train, x_test, type)
            acc = accuracy(y_test, y_predict)
            print('The test accuracy for the', data, 'dataset is', acc, '%')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import sys
sys.path.append('../')
from data.data_utils import load_dataset

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
# x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=1000, d=2)
# x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
x_train = np.vstack([x_valid, x_train])
y_train = np.vstack([y_valid, y_train])
np.random.seed(5)
np.random.shuffle(x_train)
np.random.seed(5)
np.random.shuffle(y_train)

def l1_dist(x_train, x_test):
    return np.sum(np.abs(x_train-x_test), axis=1)

def l2_dist(x_train, x_test):
    return np.sqrt(np.sum(np.square(x_train-x_test), axis=1))

def knn_regressor(x_train, y_train, x_test, k):
    dist = l2_dist(x_train, x_test) # Calculate Distance
    i_nn = np.argpartition(dist, kth=k)[:k] # Obtain Indices of Nearest Neighbours
    y_test = np.average(y_train[i_nn]) # Calculate Predicted Value by Taking Average of Neighbours
    return y_test

def knn_validate(x_train, y_train, k, v):
    y_test = []
    y_predict = []
    kf = KFold(n_splits=v)
    for train_idx, test_idx in kf.split(x_train):
        x_train_f, x_test_f = x_train[train_idx], x_train[test_idx]
        y_train_f, y_test_f = y_train[train_idx], y_train[test_idx]
        for j in range(len(x_test_f)):
            y_predict.append(knn_regressor(x_train_f, y_train_f, x_test_f[j], k))
            y_test.append(y_test_f[j])
    rmse = mean_squared_error(y_test, y_predict, squared=False)
    return rmse

def graph_loss(k, v_rmse):
    plt.figure()
    plt.plot(k, v_rmse)
    plt.xlabel('k')
    plt.ylabel('Cross-Validation RMSE')
    plt.title('k-NN Regression Algorithm Cross-Validation RMSE')
    plt.grid()
    plt.show()

def graph_test(x_test, y_test, y_predict):
    plt.figure()
    plt.plot(x_test, y_test, x_test, y_predict)
    plt.xlabel('x_test')
    plt.ylabel('y_test')
    plt.title('k-NN Regression Algorithm Test Set Prediction')
    plt.legend(['Ground Truth', 'Prediction'])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    print('k-NN Regression Algorithm\n\nPerforming 5-fold cross-validation for k = 1 to k = 30:')
    y_predict = []
    v_rmse = []
    for i in range(30):
        v_rmse.append([i+1, knn_validate(x_train, y_train, i+1, 5)])
        print('k =', v_rmse[i][0], "RMSE =", v_rmse[i][1])
    nv_rmse = np.array(v_rmse)
    graph_loss(nv_rmse[:,0], nv_rmse[:,1])
    v_rmse.sort(key=lambda x: x[1])
    k = v_rmse[0][0]
    print('\nThe estimated k value is', k, 'and the minimum cross-validation RMSE value is', v_rmse[0][1], '\n')
    print('Performing prediction on test set using chosen k value of', k, ':')
    for j in range(len(x_test)):
        y_predict.append(knn_regressor(x_train, y_train, x_test[j], k))
    t_rmse = mean_squared_error(y_test, y_predict, squared=False)
    print('The test RMSE value for k =', k, 'is', t_rmse)
    graph_test(x_test, y_test, y_predict)

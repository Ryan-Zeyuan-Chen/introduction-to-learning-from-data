# Assignment 1 - k-Nearest Neighbours (k-NN) and Linear Regression Algorithm

## Q1) k-NN Algorithm for Regression

The k-NN algorithm is implemented for regression with two different distance metrics (l<sub>2</sub> and l<sub>1</sub>). The nearest neighbours are computed using a brute-force approach. The distance metrics are implemented through `l2_dist(x_train, x_test)` and `l1_dist(x_train, x_test)`, which can be alternated in `knn_regressor(x_train, y_train, x_test, k)`. The value of `k` is estimated using 5-fold cross-validation to minimize the root-mean-square error (RMSE) loss.

The algorithm is applied to the `mauna_loa` dataset and the estimated `k` value is reported. The cross-validation RMSE and test RMSE values are presented as well.

For this model, The cross-validation loss across `k` and prediction on the test set are plotted in separate figures.

## Q2) k-NN Algorithm for Classification

The k-NN algorithm is implemented for classification with two different distance metrics (l<sub>2</sub> and l<sub>1</sub>). The nearest neighbours are computed using a brute-force approach. The distance metrics are implemented through `l2_dist(x_train, x_test)` and `l1_dist(x_train, x_test)`, which can be alternated in `knn_classifier(x_train, y_train, x_test, k)`. The value of `k` is estimated by maximizing the accuracy (fraction of correct predictions) on the validation split.

The algorithm is applied to the `iris` dataset and the estimated `k` value is reported. The validation accuracy and test accuracy are also presented.

## Q3) k-NN Algorithm using the K-D Tree Data Structure

The k-d tree data structure is utilized to compute the nearest neighbours of the k-NN regression algorithm. This approach computes the nearest neighbours for multiple test points simultaneously and is evaluated compared to the previous implemented brute-force method.

A performance study is conducted by making predictions on the test set of the `rosenbrock` regression dataset with `n_train=5000`, the l<sub>2</sub> distance metric and `k=5`. The run-times of both approaches are reported for varying values of `d` in a single plot.

## Q4) Linear Regression Algorithm

A linear regression algorithm is implemented to minimize the least-squares loss function using singular value decomposition (SVD). Both the training and validations sets are utilized to predict on the test set.

The algorithm is applied to all regression and classification datasets. The test RMSE and test accuracy are presented for regression and classification datasets respectively.

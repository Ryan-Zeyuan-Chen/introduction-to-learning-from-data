The `data_utils.py` module contains functions for loading, and viewing data.

The following code demonstrates how to load each of the datasets (note that `rosenbrock` is loaded differently since the number of training points and dimensionality must be specified manually).
```
from data_utils import load_dataset
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', n_train=5000, d=2)
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
```
Each dataset has designated training, validation and testing splits.

import numpy as np
import pandas as pd
import pickle

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show



# def make_linearY(X, s_indices, errors):
#     Y = X[:, s_indices[0]]**10 \
#         + (1 / X[:, s_indices[1]]) \
#         - np.sin(5 * X[:, s_indices[2]]) \
#         + np.exp(X[:, s_indices[3]]) \
#         + errors \
#         - np.zeros(len(X)) + X[:, s_indices[4]] < 0 \
        

#     return Y

def make_linearY(X, s_indices, errors):
    Y = np.cumsum(X[:, s_indices], axis = 1)[:,-1] + errors

    return Y

def make_nonlinearY(X, s_indices, errors):
    Y = np.cumprod(X[:, s_indices], axis = 1)[:,-1] + errors
        
    return Y

def get_mse(pred, y):
    
    bias = (1/len(pred)) * np.sum(pred-y)
    var = np.var(pred)
    
    mse = bias**2 + var
    
    return [bias, var, mse]



seeds = np.arange(0, 1e5).astype(int)
np.random.seed(seeds[0])
n_train = 1000
n_test = n_train

d = 10
s = 2
steps = np.linspace(1, 50, num=20).astype(int)



X_train = np.random.uniform(low = 0, high = 2, size=(n_train, d))
X_train = np.random.normal(loc=5, size=(n_train, d))

# X_test = np.random.binomial(1, p=0.3, size=(n_test, d))
# X_test[X_test == 0] = -3
# X_test[X_test == 1] = 3
X_test = np.random.uniform(low = -3, high = 3, size=(n_train, d))


errors_train = np.random.uniform(size=(n_train))
errors_test = np.random.uniform(size=(n_test))

s_indices = np.arange(s)

linearY_train = make_linearY(X_train, s_indices, errors_train)
linearY_test = make_linearY(X_test, s_indices, errors_test)

nonlinearY_train = make_nonlinearY(X_train, s_indices, errors_train)
nonlinearY_test = make_nonlinearY(X_test, s_indices, errors_test)


np.var(linearY_train)
np.var(nonlinearY_train)


a = np.concatenate((X_train, np.reshape(linearY_train, (n_train,1))), axis=1)
b = np.concatenate((X_train, np.reshape(nonlinearY_train, (n_train,1))), axis=1)


x_condition = 1
plot_data_lin = a[a[:, x_condition].argsort()]
plot_data_nonlin = b[b[:, x_condition].argsort()]

p_lin = figure()
p_lin.dot(plot_data_lin[:, x_condition], plot_data_lin[:,-1],
      size=20)

show(p_lin)



p_nonlin = figure()
p_nonlin.dot(plot_data_nonlin[:, x_condition], plot_data_nonlin[:,-1],
      size=20)

show(p_nonlin)
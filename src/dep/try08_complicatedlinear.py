import numpy as np
import pandas as pd
import pickle
import matplotlib
import graphviz

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn import tree

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

d = 100
s = 5
steps = np.linspace(1, 50, num=20).astype(int)



X_train = np.random.uniform(low = -3, high = 3, size=(n_train, d))
X_train = np.random.normal(loc=5, size=(n_train, d))

# X_test = np.random.binomial(1, p=0.3, size=(n_test, d))
# X_test[X_test == 0] = -3
# X_test[X_test == 1] = 3
X_test = np.random.uniform(low = -3, high = 3, size=(n_test, d))
X_test = np.random.normal(loc=5, size=(n_train, d))


errors_train = np.random.uniform(size=(n_train))
errors_test = np.random.uniform(size=(n_test))

s_indices = np.arange(s)

linearY_train = make_linearY(X_train, s_indices, errors_train)
linearY_test = make_linearY(X_test, s_indices, errors_test)

nonlinearY_train = make_nonlinearY(X_train, s_indices, errors_train)
nonlinearY_test = make_nonlinearY(X_test, s_indices, errors_test)


np.var(linearY_train)
np.var(nonlinearY_train)



params = {"random_state": 0, 
          "max_features": "auto"} 

stump_lin = DecisionTreeRegressor(splitter="best", **params)
stump_nonlin = DecisionTreeRegressor(splitter="best", **params)

stump_lin.fit(X_train, linearY_train)
stump_nonlin.fit(X_train, nonlinearY_train)



dot_data = tree.export_graphviz(stump_lin, max_depth=4)
graph = graphviz.Source(dot_data) 
graph.render("lin")

dot_data1 = tree.export_graphviz(stump_nonlin, max_depth=4)
graph = graphviz.Source(dot_data1) 
graph.render("nonlin")

stump_lin.get_depth()
stump_nonlin.get_depth()


pred_lin = stump_lin.predict(X_test)
pred_nonlin = stump_nonlin.predict(X_test)

print(get_mse(pred_lin, linearY_test))
print(get_mse(pred_nonlin, nonlinearY_test))
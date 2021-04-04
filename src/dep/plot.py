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



def make_linearY(X, s_indices, errors):
    factors = np.random.normal(size=(s))
    Y = np.empty(X.shape[0])
    Y = np.dot(X[:,s_indices], factors) + errors

    return Y

def make_nonlinearY(X, s_indices, errors):
    Y= np.cumprod(X[:, s_indices], axis = 1)[:,-1] + errors
        
    return Y

###
seeds = np.arange(0, 1e5).astype(int)
n_train = 1000
n_test = n_train

d = 20
s = 5
steps = np.linspace(1, 50, num=20).astype(int)

###
np.random.seed(seeds[1])
    
X_train = np.random.uniform(low = -3, high = 3, size=(n_train, d))

X_test = np.random.binomial(1, p=0.3, size=(n_test, d))
X_test[X_test == 0] = -3
X_test[X_test == 1] = 3


errors_train = np.random.uniform(size=(n_train))
errors_test = np.random.uniform(size=(n_test))

s_indices = np.arange(s)

linearY_train = make_linearY(X_train, s_indices, errors_train)
linearY_test = make_linearY(X_test, s_indices, errors_test)

nonlinearY_train = make_nonlinearY(X_train, s_indices, errors_train)
nonlinearY_test = make_nonlinearY(X_test, s_indices, errors_test)


data_lin = {"X_train": X_train,
        "X_test": X_test,
        "y_train": linearY_train,
        "y_test": linearY_test}


data_nonlin = {"X_train": X_train,
        "X_test": X_test,
        "y_train": nonlinearY_train,
        "y_test": nonlinearY_test}


params = {"random_state": 0, 
        "max_features": d} 

# params = {"random_state": 0, 
#           "max_features": d,
#           "min_samples_leaf": step,
#           "max_depth": 3}



stump_lin = DecisionTreeRegressor(splitter="best", **params)
stump_nonlin = DecisionTreeRegressor(splitter="best", **params)

stump_lin.fit(data_lin["X_train"], data_lin["y_train"])
stump_nonlin.fit(data_nonlin["X_train"], data_nonlin["y_train"])


dot_data = tree.export_graphviz(stump_lin, max_depth=4)
graph = graphviz.Source(dot_data) 
graph.render("linear_best") 

dot_data1 = tree.export_graphviz(stump_nonlin, max_depth=4)
graph1 = graphviz.Source(dot_data1) 
graph1.render("nonlinear_best") 


stump_lin.get_depth()
stump_nonlin.get_depth()
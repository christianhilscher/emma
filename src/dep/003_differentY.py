import numpy as np
import pandas as pd
import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, gridplot

def cond_var(dataf, bin_number=100, xcond=0):
    dataf = dataf.copy()
    
    df_out = dataf.groupby(pd.cut(dataf[xcond], 
                                  bins = bin_number))["L_Y",
                                                      "P_Y",
                                                      "N_Y"].std()
    return df_out.reset_index()

def L_Y(X, errors):
    Y = X[:,0] + X[:,1] + 2*X[:,2] + 2*X[:,3] + errors
    
    return np.ravel(Y)

def P_Y(X, errors):
    Y = np.sin(np.pi*X[:,0]) \
        + (4 / (1 + np.exp(-20 * X[:,1] + 10))) \
        + 2*X[:,2] \
        + 2*X[:,3] \
        + errors
        
    return np.ravel(Y)

def N_Y(X, errors):
    Y = np.sin(np.pi*X[:,0]) \
        + (4 / (1 + np.exp(-20 * X[:,1] + 10))) \
        + 2*X[:,2] \
        + 2*X[:,3] \
        + 3*X[:,2]*X[:,3] \
        + errors
        
    return np.ravel(Y)

def get_depth(dici, split = "best"):
    
    params = {"random_state": 0, "max_features": "auto"} 
    
    stump_L = DecisionTreeRegressor(splitter=split, **params)
    stump_P = DecisionTreeRegressor(splitter=split, **params)
    stump_N = DecisionTreeRegressor(splitter=split, **params)
    
    for m in dici.keys():
        X = dici[m].copy()
        
        L_Y = X.pop("L_Y")
        P_Y = X.pop("P_Y")
        N_Y = X.pop("N_Y")
        
        stump_L.fit(X, L_Y)
        stump_P.fit(X, P_Y)
        stump_N.fit(X, N_Y)
        
        print("------------------------------")
        print("Model: ", m)
        print("L depth: ", stump_L.get_depth())
        print("P depth: ", stump_P.get_depth())
        print("N depth: ", stump_N.get_depth())

# np.random.seed(2021)
n = 10000
d = 20

X1 = np.random.uniform(low = -5, high = 5,size = (n, d))
X2 = np.random.uniform(low = 5, high = 10, size = (n, d))
X3 = np.random.normal(loc = 0, size = (n, d))
X4 = np.random.normal(loc = 5, size = (n, d))
X5 = np.random.uniform(low=-1, high=2,size=(n, d))

X_list = [X1, X2, X3, X4, X5]

errors = np.random.normal(size = (1, n))
data = {}
plot_list = []

for (m, x) in enumerate(X_list):
    
    df_tmp = pd.DataFrame(x)
    df_tmp["L_Y"] = L_Y(x, errors)
    df_tmp["P_Y"] = P_Y(x, errors)
    df_tmp["N_Y"] = N_Y(x, errors)
    
    df_out = cond_var(df_tmp, 100)
    
    print("------------------------------")
    print("Model: ", m)
    print("Linear: ", df_out["L_Y"].std())
    print("Partial: ", df_out["P_Y"].std())
    print("Non: ", df_out["N_Y"].std())
    # plot_list.append(plot_cond_var(df_out))
    data[m] = df_tmp 
   
    
# No real difference, if anything, "best" is shallower
get_depth(data, split="random")
get_depth(data, split="best")

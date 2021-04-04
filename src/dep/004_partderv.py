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
    Y = X[:,0]**2 + X[:,1]**2 + 2*X[:,2]**2 + 2*X[:,3]**2 + errors
        
    return np.ravel(Y)

def N_Y(X, errors):
    Y = X[:,0] * X[:,1] * X[:,2] * X[:,3] + errors
        
    return np.ravel(Y)

def make_data_depth(x_list, xt_list, err, errt):
    
    data_train = {}
    data_test = {}
    
    for (m, x) in enumerate(x_list):
        
        df_tmp = pd.DataFrame(x)
        df_tmp["L_Y"] = L_Y(x, err)
        df_tmp["P_Y"] = P_Y(x, err)
        df_tmp["N_Y"] = N_Y(x, err)
        
        # df_out = cond_var(df_tmp, 100)
        
        # print("------------------------------")
        # print("Model: ", m)
        # print("Linear: ", df_out["L_Y"].std())
        # print("Partial: ", df_out["P_Y"].std())
        # print("Non: ", df_out["N_Y"].std())
        # plot_list.append(plot_cond_var(df_out))
        data_train[m] = df_tmp 
        
    for (m, x) in enumerate(xt_list):
        
        df_tmp = pd.DataFrame(x)
        df_tmp["L_Y"] = L_Y(x, errt)
        df_tmp["P_Y"] = P_Y(x, errt)
        df_tmp["N_Y"] = N_Y(x, errt)
        
        # print("------------------------------")
        # print("Model: ", m)
        # print("Linear: ", df_out["L_Y"].std())
        # print("Partial: ", df_out["P_Y"].std())
        # print("Non: ", df_out["N_Y"].std())
        # plot_list.append(plot_cond_var(df_out))
        data_test[m] = df_tmp 
        
    return data_train, data_test

def get_depth(dici):
    
    out_dici = {}
    for method in ["random", "best"]:
        
        params = {"random_state": 0, "max_features": "auto"} 
        
        stump_L = DecisionTreeRegressor(splitter=method, **params)
        stump_P = DecisionTreeRegressor(splitter=method, **params)
        stump_N = DecisionTreeRegressor(splitter=method, **params)
        
        for m in dici.keys():
            X = dici[m].copy()
            
            L_Y = X.pop("L_Y")
            P_Y = X.pop("P_Y")
            N_Y = X.pop("N_Y")
            
            stump_L.fit(X, L_Y)
            stump_P.fit(X, P_Y)
            stump_N.fit(X, N_Y)
            
            L_depth = stump_L.get_depth()
            P_depth = stump_P.get_depth()
            N_depth = stump_N.get_depth()
            
            ind = (method, m)
            out_dici[ind] = [L_depth, P_depth, N_depth]
    return out_dici

def get_bias(dici_train, dici_test):
    
    out_dici = {}
    for method in ["random", "best"]:
    
        params = {"random_state": 0, "max_features": "auto"} 
        
        stump_L = DecisionTreeRegressor(splitter=method, **params)
        stump_P = DecisionTreeRegressor(splitter=method, **params)
        stump_N = DecisionTreeRegressor(splitter=method, **params)
        
        reg_L = BaggingRegressor(base_estimator=stump_L,
                                    n_jobs=-1,
                                    n_estimators=100)
        reg_P = BaggingRegressor(base_estimator=stump_P,
                                    n_jobs=-1,
                                    n_estimators=100)
        reg_N = BaggingRegressor(base_estimator=stump_N,
                                    n_jobs=-1,
                                    n_estimators=100)

        
        for m in dici_train.keys():
            X_train = dici_train[m].copy()
            X_test = dici_test[m].copy()
            
            L_Y = X_train.pop("L_Y")
            P_Y = X_train.pop("P_Y")
            N_Y = X_train.pop("N_Y")
            
            reg_L.fit(X_train, L_Y)
            reg_P.fit(X_train, P_Y)
            reg_N.fit(X_train, N_Y)
            
            L_Y_test = X_test.pop("L_Y")
            P_Y_test = X_test.pop("P_Y")
            N_Y_test = X_test.pop("N_Y")
            
            L_bias = get_mse(reg_L.predict(X_test), L_Y_test)
            P_bias = get_mse(reg_P.predict(X_test), P_Y_test)
            N_bias = get_mse(reg_N.predict(X_test), N_Y_test)
            
            ind = (method, m)
            out_dici[ind] = [L_bias, P_bias, N_bias]
    return out_dici
        
        
def return_table(data_train, data_test, mode="depth"):
    
    if mode == "depth":
        r_dici = get_depth(data_train, split="random")
        b_dici = get_depth(data_train, split="best")
    else:
        r_dici = get_bias(data_train, data_test, split="random")
        b_dici = get_bias(data_train, data_test, split="best")
        
    df_r = pd.DataFrame(r_dici).transpose()
    df_r.columns = ["L_random", "P_random", "N_random"]
    
    df_b = pd.DataFrame(b_dici).transpose()
    df_b.columns = ["L_best", "P_best", "N_best"]
    
    df_out = pd.concat([df_r, df_b], axis=1)
    return df_out  
        
        
def get_mse(pred, y):
    
    bias = (1/len(pred)) * np.sum(pred-y)
    var = np.var(pred)
    
    mse = bias**2 + var
    
    return bias

np.random.seed(2021)
n_list = np.arange(1, 11) * 10000
d = 4

res = {}
for (ind, n) in enumerate(n_list):


    X0 = np.random.uniform(low = -5, high = 5,size = (n, d))
    X0t = np.random.uniform(low = -5, high = 5,size = (n, d))

    X1 = np.random.uniform(low = 5, high = 10, size = (n, d))
    X1t = np.random.uniform(low = 5, high = 10, size = (n, d))

    X2 = np.random.normal(loc = 0, size = (n, d))
    X2t = np.random.normal(loc = 0, size = (n, d))

    X3 = np.random.normal(loc = 5, size = (n, d))
    X3t = np.random.normal(loc = 5, size = (n, d))

    X4 = np.random.uniform(low = -100, high = 100,size=(n, d))
    X4t = np.random.uniform(low = -100, high = 100,size=(n, d))


    X_list = [X0, X2, X4]
    Xt_list = [X0t, X2t, X4t]

    errors = np.random.normal(size = (1, n))
    errorst = np.random.normal(size = (1, n))



        
    df_train, df_test = make_data_depth(X_list, Xt_list, errors, errorst)
    # pd.DataFrame(get_depth(df_train))
    res[ind] = pd.DataFrame(get_bias(df_train, df_test))
    print(n)
  

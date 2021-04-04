import numpy as np
import pandas as pd
import pickle

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, gridplot


def make_linearY(X, s_indices, errors):
    Y = np.cumsum(X[:, s_indices], axis = 1)[:,-1] + errors
        
    return np.ravel(Y)

def make_nonlinearY(X, s_indices, errors):
    Y = np.cumprod(X[:, s_indices], axis = 1)[:,-1] + errors
        
    return np.ravel(Y)

def makeX(n, d):
    
    # X = np.random.normal(loc=3, size=(n, d))
    X = np.random.uniform(low=5, high=10, size=(n, d))
    
    return X

def cond_var(dataf, bin_number=100, xcond=0):
    dataf = dataf.copy()
    
    df_out = dataf.groupby(pd.cut(dataf[xcond], 
                                  bins = bin_number))["Y_lin",
                                                      "Y_nonlin"].std()
    return df_out.reset_index()

def plot_cond_var(dataf):

    x = [element.left for element in dataf[0]]
    p_lin = figure(title="Linear model")
    p_nonlin = figure(title = "Nonlinear model")

    p_lin.line(x=x, y=dataf["Y_lin"])
    p_nonlin.line(x=x, y=dataf["Y_nonlin"])

    return [p_lin, p_nonlin]
                    
def get_depth(dici, split = "best"):
    
    params = {"random_state": 0, "max_features": "auto"} 
    
    stump_lin = DecisionTreeRegressor(splitter=split, **params)
    stump_nonlin = DecisionTreeRegressor(splitter=split, **params)
    
    for m in dici.keys():
        X = dici[m].copy()
        
        Y_lin = X.pop("Y_lin")
        Y_nonlin = X.pop("Y_nonlin")
        
        stump_lin.fit(X, Y_lin)
        stump_nonlin.fit(X, Y_nonlin)
        
        print("------------------------------")
        print("Model: ", m)
        print("Linear depth: ", stump_lin.get_depth())
        print("Nonlinear depth: ", stump_nonlin.get_depth())
           
def get_data(dici):

    forest_data = {}

    for (m, df) in zip(dici.keys(), dici.values()):
        X = df.copy()

        Y_lin = X.pop("Y_lin")
        Y_nonlin = X.pop("Y_nonlin")

        train = np.random.uniform(size=(X.shape[0])) < 0.75

        forest_dict = {"X_train": X[train],
                    "X_test": X[~train],
                    "Y_lin_train": Y_lin[train],
                    "Y_lin_test": Y_lin[~train],
                    "Y_nonlin_train": Y_nonlin[train],
                    "Y_nonlin_test": Y_nonlin[~train]}
        
        forest_data[m] = forest_dict
    return forest_data

def run_forest(data, linear=True):
    
    params = {"random_state": 0, "max_features": "auto"} 


    stump_random = DecisionTreeRegressor(splitter="random", **params)
    stump_best = DecisionTreeRegressor(splitter="best", **params)
    
    regr_random = BaggingRegressor(base_estimator=stump_random,
                                    n_jobs=-1,
                                    n_estimators=100)
    regr_best = BaggingRegressor(base_estimator=stump_best,
                                    n_jobs=-1,
                                    n_estimators=100)


    if linear:
        regr_random.fit(data["X_train"], data["Y_lin_train"])
        regr_best.fit(data["X_train"], data["Y_lin_train"])

        pred_random = regr_random.predict(data["X_test"])
        pred_best = regr_best.predict(data["X_test"])
    else:
        regr_random.fit(data["X_train"], data["Y_nonlin_train"])
        regr_best.fit(data["X_train"], data["Y_nonlin_train"])

        pred_random = regr_random.predict(data["X_test"])
        pred_best = regr_best.predict(data["X_test"])       
    
    return pred_random, pred_best

def get_mse(pred, y):
    
    bias = (1/len(pred)) * np.sum(pred-y)
    var = np.var(pred)
    
    mse = bias**2 + var
    
    return [bias, var, mse]

def get_bias(dataf, tries, linear=True):
    
    dataf = dataf.copy()
    result_array = np.empty(shape=(tries, 6))
    
    for t in range(tries):
    
        np.random.seed(t)
        if linear: 
            pred_random, pred_optimal = run_forest(dataf, linear=True)
            res_random = get_mse(pred_random, dataf["Y_lin_test"])
            res_optimal = get_mse(pred_optimal, dataf["Y_lin_test"])
        else:
            pred_random, pred_optimal = run_forest(dataf, linear=False)
            res_random = get_mse(pred_random, dataf["Y_nonlin_test"])
            res_optimal = get_mse(pred_optimal, dataf["Y_nonlin_test"])
            
        res_data = np.concatenate([res_random, res_optimal])

        result_array[t, :] = res_data
        print("Finished with try: ", t)
        
    return result_array

def results_for_models(dici, n_tries):
    dici= dici.copy()
    out_dici = {}

    for (m, df) in zip(dici.keys(), dici.values()):

        out_dici[m] = {
            "res_linear": get_bias(df, n_tries, linear=True),
            "res_nonlinear": get_bias(df, n_tries, linear=False)
            }
        
        print("Done with model ", m)
    return out_dici

def print_fractions(dici, n_tries):
    for m in dici.keys():
        
        res_linear = dici[m]["res_linear"]
        res_nonlinear = dici[m]["res_nonlinear"]
        
        frac_random_better_linear = sum(np.abs(res_linear[:,0]) < np.abs(res_linear[:,3])) / n_tries

        frac_random_better_nonlinear = sum(np.abs(res_nonlinear[:,0]) < np.abs(res_nonlinear[:,3])) / n_tries
        
        print("-----------------------------")
        print("Model: ", m)
        print("Frac. random better in linear regime: ", frac_random_better_linear)
        print("Frac. random better in nonlinear regime: ", frac_random_better_nonlinear)
###############################################################################

# np.random.seed(2021)
n = 10000
d = 50
s = 10

X1 = np.random.uniform(low = -5, high = 5,size = (n, d))
X2 = np.random.uniform(low = 5, high = 10, size = (n, d))
X3 = np.random.normal(loc = 0, size = (n, d))
X4 = np.random.normal(loc = 5, size = (n, d))
X5 = np.random.uniform(low=-1, high=2,size=(n, d))

X_list = [X1, X2, X3, X4, X5]

errors = np.random.normal(size = (1, n))
s_ind = np.arange(s)

data = {}
plot_list = []

for (m, x) in enumerate(X_list):
    
    df_tmp = pd.DataFrame(x)
    df_tmp["Y_lin"] = make_linearY(x, s_ind, errors)
    df_tmp["Y_nonlin"] = make_nonlinearY(x, s_ind, errors)
    
    df_out = cond_var(df_tmp, 100)
    
    print("------------------------------")
    print("Model: ", m)
    print("Linear: ", df_out["Y_lin"].std())
    print("Nonlinear: ", df_out["Y_nonlin"].std())
    plot_list.append(plot_cond_var(df_out))
    data[m] = df_tmp 

p = gridplot(plot_list)
show(p)
     
get_depth(data, split="random")
get_depth(data, split="best")

models_data = get_data(data)

n_tries = 1
abc = results_for_models(models_data, n_tries)



print_fractions(abc, n_tries)

df = abc[0]
df["res_linear"]
df["res_nonlinear"]

sum(np.abs(df["res_linear"][:, 0]) > np.abs(df["res_linear"][:, 3]))
sum(np.abs(df["res_nonlinear"][:, 2]) > np.abs(df["res_nonlinear"][:, 5]))
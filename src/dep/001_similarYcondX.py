import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, gridplot


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

def run_forest(data, step):
    
    params = {"random_state": 0, 
            "max_features": d,
            "min_samples_leaf": step} 


    stump_random = DecisionTreeRegressor(splitter="random", **params)
    stump_best = DecisionTreeRegressor(splitter="best", **params)
    
    regr_random = BaggingRegressor(base_estimator=stump_random,
                                    n_jobs=-1,
                                    n_estimators=100)
    regr_best = BaggingRegressor(base_estimator=stump_best,
                                    n_jobs=-1,
                                    n_estimators=100)


    regr_random.fit(data["X_train"], data["y_train"])
    regr_best.fit(data["X_train"], data["y_train"])

    pred_random = regr_random.predict(data["X_test"])
    pred_best = regr_best.predict(data["X_test"])
    
    return pred_random, pred_best

def wrap_try(data, result_array, t):
    res_random = np.empty(shape=(len(steps), 3))
    res_best = np.empty_like(res_random)

    for (ind, step) in enumerate(steps):
        

        pred_random, pred_best = run_forest(data, step)

        res_random[ind] = get_mse(pred_random, data["y_test"])
        res_best[ind] = get_mse(pred_best, data["y_test"])
        print("Done with step ", ind)


    res_data = np.concatenate([res_random, res_best], axis=1)
    result_array[:,:,t] = res_data
    
    return result_array


def plot(result_arr, lin):
        
    res = pd.DataFrame(data= result_arr,
                    columns=["bias_random", "var_random", "mse_random",
                                "bias_best", "var_best", "mse_best"])
    res["steps"] = steps/n_train


    variables = ["bias", "var", "mse"]
    titles = ["Bias", "Variance", "MSE"]
    ps = []

    for (ind,v) in enumerate(variables):
        
        title = titles[ind]
        p = figure(title = title)
        
        source = ColumnDataSource(res)
        
        y_random = v + "_random"
        y_best = v + "_best"
        
        p.line(x="steps", y=y_random, source=source,
            line_color = "black", line_dash="solid", line_width=3,
            legend_label = "random splits")
        p.line(x="steps", y=y_best, source=source,
            line_color = "black", line_dash="dashed", line_width=3,
            legend_label = "optimal splits")
        
        p.xaxis.axis_label="Minimum fraction left in leaf"
        
        ps.append(p)
        
    if lin:
        title = "Linear Model"
    else:
        title = "Nonlinear Model"
    grid = gridplot([ps])
    show(grid)


def run_tries(tries, steps):
    
    final_results = {"res_linear": np.zeros((len(steps), 6, tries)),
                 "res_nonlinear": np.zeros((len(steps), 6, tries))}

    for t in np.arange(tries):
        
        np.random.seed(seeds[t])
        
        X_train = makeX(n_train, d)
        X_test = makeX(n_test, d)
        
        
        errors_train = np.random.uniform(size=(n_train))
        errors_test = np.random.uniform(size=(n_test))

        s_indices = np.arange(s)

        linearY_train = make_linearY(X_train, s_indices, errors_train)
        linearY_test = make_linearY(X_test, s_indices, errors_test)

        nonlinearY_train = make_nonlinearY(X_train, s_indices, errors_train)
        nonlinearY_test = make_nonlinearY(X_test, s_indices, errors_test)

        ways = ["linear", "non_linear"]
        for funcs in ways:
            
            if funcs=="linear":

                data = {"X_train": X_train,
                        "X_test": X_test,
                        "y_train": linearY_train,
                        "y_test": linearY_test}
                
                final_results["res_linear"] = wrap_try(data, final_results["res_linear"], t)
            else:

                data = {"X_train": X_train,
                        "X_test": X_test,
                        "y_train": nonlinearY_train,
                        "y_test": nonlinearY_test}
                
                final_results["res_nonlinear"] = wrap_try(data, final_results["res_nonlinear"], t)
            
        print("Done with try", t)
        
    return final_results

def plot_YcondX(X, Y_lin, Y_nonlin, cond=0):
    
    lin_mat = np.concatenate((X, Y_lin), axis=1)
    nonlin_mat = np.concatenate((X, Y_nonlin), axis=1)
    
    plot_data_lin = lin_mat[lin_mat[:, cond].argsort()]
    plot_data_nonlin = nonlin_mat[nonlin_mat[:, cond].argsort()]
    
    p_lin = figure()
    p_lin.dot(plot_data_lin[:, cond], plot_data_lin[:,-1],
      size=20, color="blue")

    p_nonlin = figure()
    p_nonlin.dot(plot_data_nonlin[:, cond], plot_data_nonlin[:,-1],
        size=20, color="red")
    
    f = gridplot([[p_lin, p_nonlin]])
    show(f)

def makeX(n, d):
    
    # X = np.random.normal(loc=3, size=(n, d))
    X = np.random.uniform(low=5, high=10, size=(n, d))
    
    return X




seeds = np.arange(0, 1e5).astype(int)
n_train = 10000
n_test = n_train

d = 100
s = 5



t1 = makeX(1000, d)
err = np.zeros(1000)
sind = np.arange(s)

a = make_linearY(t1, sind, err)
b = make_nonlinearY(t1, sind, err)


a1 = a.reshape(len(a), 1)
b1 = b.reshape(len(b), 1)

plot_YcondX(t1, a1, b1, cond = 1)


steps = [1]
tries = 2

res = run_tries(tries, steps)


linear = res["res_linear"]
nonlinear = res["res_nonlinear"]


print("linear: ")
sum(sum(np.abs(linear[:,0,:]) > np.abs(linear[:,3,:])))/tries

print("nonlinear:")
sum(sum(np.abs(nonlinear[:,0,:]) > np.abs(nonlinear[:,3,:])))/tries


# 0.45 vs. 0.66 - np.random.normal(loc=5)
# 0.485 vs 0.41 - np.random.normal(loc=0)
params = {"random_state": 0, 
          "max_features": "auto"} 

stump_lin = DecisionTreeRegressor(splitter="best", **params)
stump_nonlin = DecisionTreeRegressor(splitter="best", **params)

stump_lin.fit(t1, a)
stump_nonlin.fit(t1, b)



dot_data = tree.export_graphviz(stump_lin, max_depth=4)
graph = graphviz.Source(dot_data) 
graph.render("lin")

dot_data1 = tree.export_graphviz(stump_nonlin, max_depth=4)
graph = graphviz.Source(dot_data1) 
graph.render("nonlin")

stump_lin.get_depth()
stump_nonlin.get_depth()
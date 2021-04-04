import numpy as np
import pandas as pd
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, gridplot

def get_mse(pred, y):
    
    bias = (1/len(pred)) * np.sum(pred-y)
    var = np.var(pred)
    
    mse = bias**2 + var
    
    return [bias, var, mse]

def run_forest(data, step):
    # params = {"max_features": d,
    #         "min_samples_leaf": step}
    
    params = {"random_state": 0, 
            "max_features": 0.5,
            "min_samples_leaf": step} 
    
    # params = {"random_state": 0, 
    #           "max_features": d,
    #           "min_samples_leaf": step,
    #           "max_depth": 10}


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


    res_data = np.concatenate([res_random, res_best], axis=1)
    result_array[:,:,t] = res_data
    
    return result_array


def plot(result_arr):
        
    res = pd.DataFrame(data= result_arr,
                    columns=["bias_random", "var_random", "mse_random",
                                "bias_best", "var_best", "mse_best"])
    res["steps"] = steps


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
        
        p.xaxis.axis_label="Samples left in leaf"
        
        ps.append(p)
        
    grid = gridplot([ps])
    show(grid)

     

# np.random.seed(2021)
n = 15000
d = 100
s = 5
steps = np.unique(np.round(2 ** np.arange(0, 9, 0.3))).astype(int)
tries = 1
final_results = {"res_linear": np.zeros((len(steps), 6, tries)),
                 "res_nonlinear": np.zeros((len(steps), 6, tries))}

for t in np.arange(tries):
    X = np.random.uniform(size=(n, d))
    # X = np.random.binomial(1, p=0.3, size=(n, d))

    s_indices = np.random.randint(0, d, s)
    factors = np.random.normal(size=(s))
    # factors = np.arange(10, 60, 10)

    # X[:,s_indices[0]] = np.zeros(n)


    errors = np.random.uniform(size=(n))
    # errors = np.zeros(n)

    Y1 = np.empty(X.shape[0])
    Y1 = np.dot(X[:,s_indices], factors) + errors

    Y2 = X[:, s_indices[0]] + 3 * X[:, s_indices[1]]*X[:, s_indices[2]] \
        - 0.4 * (X[:, s_indices[3]]*X[:, s_indices[4]])**2 \
        + errors

    ways = ["linear", "non_linear"]
    for funcs in ways:
        
        if funcs=="linear":
            X_train, X_test, y_train, y_test = train_test_split(X, Y1, 
                                                                test_size = 0.15,
                                                                random_state=0)

            data = {"X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test}
            
            final_results["res_linear"] = wrap_try(data, final_results["res_linear"], t)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, Y2, 
                                                                test_size = 0.15,
                                                                random_state=0)

            data = {"X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test}
            
            final_results["res_nonlinear"] = wrap_try(data, final_results["res_nonlinear"], t)
        
    print("Done with try", t)
        


with open('linear.pkl','wb') as f:
    pickle.dump(final_results["res_linear"], f)
    
with open('nonlinear.pkl','wb') as f:
    pickle.dump(final_results["res_nonlinear"], f)


tr = 0
lin_results = final_results["res_linear"][:,:, tr]
plot(lin_results)

nonlin_results = final_results["res_nonlinear"][:,:, tr]
plot(nonlin_results)


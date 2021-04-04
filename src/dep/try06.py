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
    factors = np.random.normal(size=(s))
    Y = np.empty(X.shape[0])
    Y = np.dot(X[:,s_indices], factors) #+ errors

    return Y

def make_nonlinearY(X, s_indices, errors):
    Y= np.cumprod(X[:, s_indices], axis = 1)[:,-1] #+ errors
        
    return Y

def get_mse(pred, y):
    
    bias = (1/len(pred)) * np.sum(pred-y)
    var = np.var(pred)
    
    mse = bias**2 + var
    
    return [bias, var, mse]

def run_forest(data, step):
    # params = {"max_features": d,
    #         "min_samples_leaf": step}
    
    params = {"random_state": 0, 
            "max_features": d,
            "min_samples_leaf": step} 
    
    # params = {"random_state": 0, 
    #           "max_features": d,
    #           "min_samples_leaf": step,
    #           "max_depth": 3}


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

     

seeds = np.arange(0, 1e5).astype(int)
n_train = 10000
n_test = n_train

d = 20
s = 4
steps = np.linspace(1, 50, num=20).astype(int)
# steps = np.unique(np.round(np.linspace(1, 10, 50)**2)).astype(int)
steps = [1]

tries = 1
final_results = {"res_linear": np.zeros((len(steps), 6, tries)),
                 "res_nonlinear": np.zeros((len(steps), 6, tries))}

for t in np.arange(tries):
    
    # np.random.seed(seeds[t])
    
    X_train = np.random.uniform(low = -3, high = 3, size=(n_train, d))
    X_train = np.random.normal(loc=5, size=(n_train, d))
    
    # X_test = np.random.binomial(1, p=0.3, size=(n_test, d))
    # X_test[X_test == 0] = -3
    # X_test[X_test == 1] = 3
    X_test = np.random.uniform(low = -3, high = 3, size=(n_train, d))
    X_test = np.random.normal(loc=5, size=(n_train, d))
    
    
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
        
print(final_results["res_linear"][0][[0, 3]])
print(final_results["res_nonlinear"][0][[0, 3]])

# with open('linear.pkl','wb') as f:
#     pickle.dump(final_results["res_linear"], f)
    
# with open('nonlinear.pkl','wb') as f:
#     pickle.dump(final_results["res_nonlinear"], f)


# fileo = open('linear.pkl', 'rb')
# # loading data
# linear = pickle.load(fileo)
# # close the file
# fileo.close()

# fileo = open('nonlinear.pkl', 'rb')
# # loading data
# nonlinear = pickle.load(fileo)
# # close the file
# fileo.close()


tr = 0
lin_results = final_results["res_linear"][:,:, tr]
plot(lin_results, lin=True)


nonlin_results = final_results["res_nonlinear"][:,:, tr]
plot(nonlin_results, lin=False)

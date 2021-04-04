import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, gridplot


# np.random.seed(2021)


def get_mse(pred, y):
    
    bias = (1/len(pred)) * np.sum(pred-y)
    var = np.var(pred)
    
    mse = bias**2 + var
    
    return [bias, var, mse]


n = 10000
d = 20
s = 5
steps = np.linspace(1e-15, 1e-1, 100)
tries = 100

final_results = np.zeros((len(steps), 6, tries))

for t in np.arange(tries):

    X = np.random.normal(size=(n, d))

    s_indices = np.random.randint(0, d, s)
    factors = np.random.normal(size=(s))
    # factors = np.arange(10, 60, 10)

    # X[:,s_indices[0]] = np.zeros(n)


    errors = np.random.uniform(size=(n))
    # errors = np.zeros(n)

    Y1 = np.empty(X.shape[0])
    Y1 = np.dot(X[:,s_indices], factors) + errors

    Y2 = X[:, s_indices[0]]**2 \
        + (1/ (X[:, s_indices[1]])) * np.sin(5 * X[:, s_indices[2]]) \
        - np.exp(X[:, s_indices[3]]) * X[:, s_indices[4]] \
        + errors


    X_train, X_test, y_train, y_test = train_test_split(X, Y2, 
                                                        test_size = 0.3)

    data = {"X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test}



    # steps = steps.astype(int)
    res_random = np.empty(shape=(len(steps), 3))
    res_best = np.empty_like(res_random)

    for (ind, step) in enumerate(steps):
        
        params = {"max_features": d,
                  "min_samples_leaf": step}
        
        # params = {"random_state": 0, 
        #         "max_features": d,
        #         "min_samples_leaf": step} 
        
        # params = {"random_state": 0, 
        #           "max_features": d,
        #           "min_samples_leaf": step,
        #           "max_depth": 10}


        stump_random = DecisionTreeRegressor(splitter="random", **params)
        stump_best = DecisionTreeRegressor(splitter="best", **params)
        
        regr_random = BaggingRegressor(base_estimator=stump_random,
                                        n_jobs=-1)
        regr_best = BaggingRegressor(base_estimator=stump_best,
                                      n_jobs=-1)


        regr_random.fit(data["X_train"], data["y_train"])
        regr_best.fit(data["X_train"], data["y_train"])

        pred_random = regr_random.predict(data["X_test"])
        pred_best = regr_best.predict(data["X_test"])


        res_random[ind] = get_mse(pred_random, data["y_test"])
        res_best[ind] = get_mse(pred_best, data["y_test"])
        # print("Done with step", ind)


    res_data = np.concatenate([res_random, res_best], axis=1)
    final_results[:,:,t] = res_data
    print("Done with ", t)
    

final_results = np.abs(final_results)
results = np.mean(final_results, axis=2)
res = pd.DataFrame(data= results,
                   columns=["bias_random", "var_random", "mse_random",
                            "bias_best", "var_best", "mse_best"])

res["steps"] = steps
res

for a in np.arange(final_results.shape[0]):
    prop = final_results[a, 0,:] > final_results[a, 3,:]
    print(sum(prop)/len(prop))
   


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
    
    p.xaxis.axis_label="Fraction left in leaf"
    
    ps.append(p)
    
grid = gridplot([ps])
show(grid)


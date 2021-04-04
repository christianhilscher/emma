import numpy as np
import pandas as pd
import pickle

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show, gridplot

def plot(result_arr, lin):
        
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
        
    if lin:
        title = "Linear Model"
    else:
        title = "Nonlinear Model"
    grid = gridplot([ps])
    show(grid)

fileo = open('linear.pkl', 'rb')
# loading data
linear = pickle.load(fileo)
# close the file
fileo.close()

fileo = open('nonlinear.pkl', 'rb')
# loading data
nonlinear = pickle.load(fileo)
# close the file
fileo.close()

steps = np.linspace(1, 50).astype(int)
res = pd.DataFrame(data= linear[:,:,0],
                    columns=["bias_random", "var_random", "mse_random",
                                "bias_best", "var_best", "mse_best"])

linear = np.abs(linear)
# According to my thinking, bias w/ optimal splitting should be somewhat 
# smaller. This is true if values here are larger than 0.5
for a in np.arange(linear.shape[0]):
    prop = linear[a, 0,:] > linear[a, 3,:]
    print(sum(prop)/len(prop))
 
 
# According to my thinking, bias w/ optimal splitting should be 
# higher here. This is true if values here are smaller than 0.5   
nonlinear = np.abs(nonlinear)

for a in np.arange(nonlinear.shape[0]):
    prop = nonlinear[a, 0,:] > nonlinear[a, 3,:]
    print(sum(prop)/len(prop))

# Printnig averages
n = len(linear[0,0,:])
lin_avg = np.zeros(linear.shape[0])
nonlin_avg = np.zeros_like(lin_avg)

for a in np.arange(linear.shape[0]):
    lin_avg[a] = sum(linear[a, 0,:] > linear[a, 3,:])/n
    nonlin_avg[a] = sum(nonlinear[a, 0,:] > nonlinear[a, 3,:])/n
    
# Shows proportion of tries where optimal splitting was better than random splitting
d = {"lin": lin_avg, "non_lin": nonlin_avg}
pd.DataFrame(data = d)



tr = 168
lin_results = linear[:,:, tr]
plot(lin_results, lin=True)

nonlin_results = nonlinear[:,:, tr]
plot(nonlin_results, lin=False)
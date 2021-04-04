import numpy as np


n = 100000000
x1 = np.random.uniform(low = 1, high = 2, size=(n))
x2 = np.random.uniform(low = 1, high = 2, size=(n))

y_lin = x1 + 3*x2
y_nonlin = x1**2 + x2**2 * 3

y_nonlin

tries = 50
var_lin = np.empty(tries)
var_nonlin = np.empty(tries)


for i in np.arange(tries):

    r1 = 0.3 / i
    r2 = 0.1 / i

    rpoint = np.random.uniform(low = 1, high = 2)


    x1_lb = rpoint - r1
    x1_ub = rpoint + r1
    x2_lb = rpoint - r2
    x2_ub = rpoint + r2


    x1_cond = (x1 > x1_lb) & (x1 < x1_ub)
    x2_cond = (x2 > x2_lb) & (x2 < x2_ub)

    y_tmp_lin = y_lin[x1_cond & x2_cond]
    y_tmp_nonlin = y_nonlin[x1_cond & x2_cond]

    var_lin[i] = np.var(y_tmp_lin)
    var_nonlin[i] = np.var(y_tmp_nonlin)

sum(var_lin < var_nonlin) / tries

sum(var_lin[:-1] > np.roll(var_lin, -1)[:-1]) / tries
sum(var_nonlin[:-1] > np.roll(var_nonlin, -1)[:-1]) / tries

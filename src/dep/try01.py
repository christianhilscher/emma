import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


np.random.seed(2021)

#############################################################################

def linear_Y(X, d, s):
    X = X.copy()
    
    s_indices = np.random.randint(0, d, s)
    factors = np.random.normal(size=(s))
    
    Y = np.empty(X.shape[0])
    Y = np.dot(X[:,s_indices], factors)
    
    return Y

def run_forest(data, ne):

    regr = RandomForestRegressor(n_estimators=ne, 
                                 random_state=0)
    
    regr.fit(data["X_train"], data["y_train"])

    pred = regr.predict(data["X_test"])

    bias = np.sum(pred-data["y_test"]) / len(data["y_test"])
    var = np.var(pred)
    
    return bias, var

def make_data(X, Y, sample_size):
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                        test_size = sample_size)

    data = {"X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test}
    return data



n = 1000
d = 20
s = 5

X = np.random.normal(size=(n, d))
Y1 = linear_Y(X, d, s)

ends = np.arange(1, 50)
res = np.empty(shape=(len(ends), 3))

for (ind, end) in enumerate(ends):

    data = make_data(X, Y1, 0.1)
    b, v = run_forest(data, end)
    mse = b**2 + v
    
    res[ind] = [b, v, mse]

    
print(res)



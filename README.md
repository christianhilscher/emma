# EMMA
Code for Master Thesis. 
Handed in on September 6th, 2021.

## Idea
A random forest works relatively well in large sample settings. 
As all data is finite, a regression tree leaves the realm of asymptotic analysis with each split and takes a step towards a finite sample setting. 
This has the following effects:

* In a large sample framework, the CART algorithm can differentiate well between signal and noise
* With every split the tree exlpoits as much signal as possible, leading to the signal strength decreasing in nodes deeper down the tree
* At the same time, due to the reduced sample size the presence of error terms is noticeable and the tree cannot differentiate well between signal and noise anymore

Introducing weights into the splitting rule of the CART algorithm takes these effects into account and leads to better predictions.
Especially when the weights are an increasing function of tree depth, sizeable gains are realized.


## Implementation
<!-- The base of the code is taken from https://liorsinai.github.io/coding/2020/12/14/random-forests-jl.html.  -->
Adaptions from a standard Random Forest Classifier include the switch form a classifier to a regressor and the introduction of weights. 
In addition, tuning parameters such as minimal observations in a leaf and maximum depth are incorporated. 
The random forest is a collection of decision tree classifiers and therefore a multi-threading is implemented to speed up calculations.

## Structure 
This repository has three folders: _src_ holds the code for the random forest estimator with the weighted splitting rule. 
The folder _figures_ is the collection of all code as well as graphs used in the paper. 
Each of those files has in the first line a comment indicating which figure it is used for. 
Additionally, the folder _data_ contains all the data generated from the files in _figures_ and then used for plotting.


### src

The random forest estimator is implemented as follows:

* DTRegressor:

    Basis of estimator, stands for Decision Tree Regressor. 
    The core of this object is a structure called _binary tree_, which is a collection of lists.
    The DTRegressor is the implementation of a single decision tree, making splits according to the weighted splitting rule.
    Setting tuning parameters such as maximum depth or minimum amount of observations within a node is possible.

* RFR:

    Stands for Random Forest Regressor and is a collection of DTRegressor trees.
    Parameter values set here passed on to all decision trees. 
    Since the individual trees are independent of each other, the estimator uses multi-threading to speed up calculations.

* Cross-Val:

    File holding the cross validation object. 
    Where for RFR and DTRegressor only one parameter value is allowed, this object can take in a vector of values for one parameter. 
    It fits the data to all possible combinations of values and implements a RFR for each of them.
    Then, it is possible to return those parameter values which minimize either bias, variance or MSE.

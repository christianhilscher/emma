# EMMA
Code for Master Thesis

## Idea
A random forest works relatively well in large sample settings. As all data is finite, a regression tree leaves the realm of asymptotic analysis with each split and takes a step towards a finite sample setting. This has the following effects:

* In a large sample framework, the CART algorithm can differentiate well between signal and noise
* With every split the tree exlpoits as much signal as possible, leading to the signal strength decreasing in nodes deeper down the tree
* At the same time, due to the reduced sample size the presence of error terms is noticeable and the tree cannot differentiate well between signal and noise anymore

Introducing weights into the splitting rule of the CART algorithm takes these effects into account and leads to better predictions.
Especially when the weights are an increasing function of tree depth, sizeable gains are realized.

## Implementation
The base of the code is taken from https://liorsinai.github.io/coding/2020/12/14/random-forests-jl.html. Adaptions include the switch form a classifier to a regressor and the introduction of weights. In addition, tuning parameters such as minimal observations in a leaf and maximum depth are incorporated. The random forest is a collection of decision tree classifiers and therefore a multi-threading is also implemented to speed up calculations.

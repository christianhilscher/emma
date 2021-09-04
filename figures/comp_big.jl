# File for generating data for comparison table

using Pkg
using Random, Distributions
using Statistics
using ProgressMeter
using DataFrames
using JLD2

include("/home/christian/UniMA/EMMA/src/RFR.jl")
include("/home/christian/UniMA/EMMA/src/cross_val.jl")
include("/home/christian/UniMA/EMMA/src/aux_functions.jl")


function big_comp(cv_list, n_list, func="friedman", res=nothing)
    
    if res===nothing
        # Initiate output DataFrame
        res = DataFrame()
    end
    
    approaches = ["CART", "weighted", "max depth", "min samples leaf"]

    for n1 in n_list
        x_train, x_test, y_train, y_test = make_data(n1, d, func, σ)

        for (ind, cv) in enumerate(cv_list)
            fit!(cv, x_train, y_train)
            rf = best_model(cv, "mse")
            pred = predict(rf, x_test)
            mse_arr = get_mse(pred, y_test)


            res = vcat(res, DataFrame(bias=mse_arr[1], 
                                        variance=mse_arr[2],
                                        mse=mse_arr[3],
                                        Approach=approaches[ind],
                                        n=n1))
            
        end
    end

    return res
end

### Make comparison
d = 10
σ = 1
m_features = d
cv_parameter = collect(1:5:31)

# Making dictionaries for different approaches
d_base = Dict{Symbol, Vector{Float64}}(
    :max_features => [m_features],
    :n_trees => [30])

d0 = copy(d_base)
d0[:α] = [0.0]

d1 = copy(d_base)
d1[:α] = cv_parameter

d2 = copy(d_base)
d2[:max_depth] = cv_parameter[2:end] # Have to start one above, otherwise depth approach does not work

d3 = copy(d_base)
d3[:min_samples_leaf] = cv_parameter

# Initializing cross validation object for different approaches
cv0 = cross_val(d0, random_state=0)
cv1 = cross_val(d1, random_state=0)
cv2 = cross_val(d2, random_state=0)
cv3 = cross_val(d3, random_state=0)


# List of observations
n_list = [250, 500, 1000, 2000, 4000, 8000, 16000]
# List of CV objects
cv_list = [cv0, cv1, cv2, cv3]
# Repetitions
reps = 100


## Make Friedman simulation
res_friedman = big_comp(cv_list, n_list, "friedman")

for i in 1:reps
    res_friedman = big_comp(cv_list, n_list, "friedman", res_friedman)
    println("\n Done with round", i)
end

jldsave("data/comp_friedman.jld2"; res)

## Make DP3 simulation
res_dp3 = big_comp(cv_list, n_list, "dp3")

for i in 1:reps
    res_dp3 = big_comp(cv_list, n_list, "dp3", res_dp3)
    println("\n Done with round", i)
end

jldsave("data/comp_dp3.jld2"; res_dp3)

## Make DP8 simulation
res_dp8 = big_comp(cv_list, n_list, "dp8")

for i in 1:reps
    res_dp8 = big_comp(cv_list, n_list, "dp8", res_dp8)
    println("\n Done with round", i)
end

jldsave("data/comp_dp8.jld2"; res)

## Make robot simulation
res_robot = big_comp(cv_list, n_list, "robot")

for i in 1:reps
    res_robot = big_comp(cv_list, n_list, "robot", res_robot)
    println("\n Done with round", i)
end

jldsave("data/comp_robot.jld2"; res)
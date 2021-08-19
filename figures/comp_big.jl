using Pkg
using Random, Distributions
using Statistics
using ProgressMeter
using DataFrames
using JLD2

import Cairo, Fontconfig

include("/home/christian/UniMA/EMMA/src/RFR.jl")
include("/home/christian/UniMA/EMMA/src/cross_val.jl")
include("/home/christian/UniMA/EMMA/src/aux_functions.jl")


function big_comp(cv_list, n_list, res=nothing)
    
    if res===nothing
        # Initiate output DataFrame
        res = DataFrame()
    end
    
    approaches = ["CART", "weighted", "max depth", "min samples leaf"]

    @showprogress for n1 in n_list
        x_train, x_test, y_train, y_test = make_data(n1, d, "friedman", σ)

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

cv0 = cross_val(d0, random_state=0)
cv1 = cross_val(d1, random_state=0)
cv2 = cross_val(d2, random_state=0)
cv3 = cross_val(d3, random_state=0)



n_list = [250, 500, 1000, 2000, 4000, 8000, 16000]
cv_list = [cv0, cv1, cv2, cv3]


# Load old dataframe to add runs
load_dict = load("data/comp_big.jld2")
res = load_dict["res"]

reps = 10
for i in 1:reps
    res = big_comp(cv_list, n_list, res)
end


# Uncomment to update old results by adding new ones
# jldsave("data/comp_big.jld2"; res)
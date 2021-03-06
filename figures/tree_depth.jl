# File for getting the data for Figure 4

using Pkg
using Statistics
using Random, Distributions
using SparseGrids
using BenchmarkTools
using DataFrames
using JLD2

include("/home/christian/UniMA/EMMA/src/RFR.jl")
include("/home/christian/UniMA/EMMA/src/cross_val.jl")
include("/home/christian/UniMA/EMMA/src/aux_functions.jl")

import Cairo, Fontconfig

# Set seed for reproducability
Random.seed!(68151)

# Set parameters
n = 2000
d = 10
σ = 1
m_features = d

# Make dictionaries for CART and weighted approach
d1 = Dict{Symbol, Vector{Float64}}(
    :max_features => [m_features],
    :n_trees => [30])

d2 = copy(d1)
d2[:α] = [0] # CART has α=0

# List of observations to go through
n_list = [250, 500, 1000, 2000, 4000, 8000, 16000]
res_mat = zeros(length(n_list), 2)


for (ind,n1) in enumerate(n_list)
    x_train, x_test, y_train, y_test = make_data(n1, d, "friedman", σ)
    cv = cross_val(d1, random_state=0)
    fit!(cv, x_train, y_train)

    # Get optimal parameter for α
    rf_α = best_model(cv, "mse")
    rf0 = RFR(param_dict = d2)

    fit!(rf_α, x_test, y_test)
    fit!(rf0, x_test, y_test)

    # Save average depth of all trees in forest
    res_mat[ind, 1] = average_depth(rf0)
    res_mat[ind, 2] = average_depth(rf_α)
    println(ind)
end

# Make dataframe and save data
df_tmp = DataFrame(res_mat, :auto)
rename!(df_tmp, ["unweighted", "weighted"])
df_tmp[!, :n] = n_list

jldsave("data/tree_depth.jld2"; df_tmp)



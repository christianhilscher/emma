using Pkg
using Statistics
using Random, Distributions
using Plots
using SparseGrids
using BenchmarkTools

include("RFR.jl")
include("cross_val.jl")
include("aux_functions.jl")


function validate_model(rf1::RFR, rf2::RFR, X::Matrix, Y::Matrix)
    
    fit!(rf1, x_test, y_test)
    fit!(rf2, x_test, y_test)
    
    pred1 = predict(rf0, X)
    pred2 = predict(rf_best, X)

    return pred1, pred2
end


# Random.seed!(68159)
n = 2000
d = 10

a_list = collect(LinRange(0, 30, 31))

d1 = Dict{Symbol, Vector{Float64}}(
    :max_features => [d],
    :n_trees => [30])

d_alpha = copy(d1)
d_alpha[:α] = a_list

d_min = copy(d1)
d_min[:min_samples_leaf] = a_list


cv_alpha = cross_val(d_alpha, random_state = 0)
cv_min = cross_val(d_min, random_state = 0)


n_runs = 100
res_mat = zeros(n_runs, 6)

for i in 1:n_runs
    x_train, x_test, y_train, y_test = make_data(n, d, "friedman", 1)
    fit!(cv_alpha, x_train, y_train, nfolds=3)
    fit!(cv_min, x_train, y_train, nfolds=3)

    rf_alpha = best_model(cv_alpha, "mse")
    rf_min = best_model(cv_min, "mse")

    pred_alpha = predict(rf_alpha, x_test)
    pred_min = predict(rf_min, x_test)

    res_mat[i, 1:3] .= get_mse(pred_alpha, y_test)
    res_mat[i, 4:end] .= get_mse(pred_min, y_test)

end

mean(res_mat[:,1])
mean(res_mat[:, 4])
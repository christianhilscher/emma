using Pkg
using Statistics
using Random, Distributions
using Plots
using SparseGrids
using BenchmarkTools

include("RFR.jl")
include("cross_val.jl")
include("aux_functions.jl")


function validation_forest(cv::cross_val)
    d_best = best_model(cv, "mse").param_dict
    
    # Taking same results as for adapted and only changing α
    d0 = copy(d_best)
    d0[:α] = 0.0
    d0[:n_features] = d

    rf0 = RFR(param_dict = d0)
    rf_best = RFR(param_dict = d_best)

    return rf0, rf_best
end

function validate_model(rf0::RFR, rf_best::RFR, X::Matrix, Y::Matrix)
    
    fit!(rf0, x_test, y_test)
    fit!(rf_best, x_test, y_test)
    
    pred0 = predict(rf0, X)
    pred_best = predict(rf_best, X)

    return pred0, pred_best
end

function get_mse(pred, y)
    bias = abs(mean(pred .- y))
    variance = var(pred)
    mse = mean((pred .- mean(y)).^2)

    return bias, variance, mse
end

Random.seed!(68159)
n = 2000
d = 10


x_train, x_test, y_train, y_test = make_data(n, d, "friedman", 1)

a_list = collect(LinRange(0, 30, 31))

d1 = Dict{Symbol, Vector{Float64}}(
    :max_features => [d],
    :n_trees => [30])

d_cv = copy(d1)
d_cv[:α] = a_list


cv1 = cross_val(d_cv, random_state = 0)
fit!(cv1, x_train, y_train, nfolds=3)

rf0, rf_best = validation_forest(cv1)
pred0, pred_best = validate_model(rf0, rf_best, x_test, y_test)

pl0 = combined_splitpoints(rf0)
pl_best = combined_splitpoints(rf_best)


seq_res = Array{Float64}(undef, length(cv1.regressor_list))
for (ind, rf) in enumerate(cv1.regressor_list)
    seq_res[ind] = average_depth(rf)
    # seq_res[ind] = strong_selection_freq(rf, 5)
end


println("Average depth: ", average_depth(cv1.regressor_list[4]))
println("Correlation between depth and bias: ", cor(cv1.bias_list, seq_res))
println("Best mse: ", best_model(cv1, "mse").param_dict, "\n")


println("Standard model: ", round.(get_mse(pred0, y_test), digits=5))
println("Adapted model: ", round.(get_mse(pred_best, y_test), digits=5))

res = get_mse(pred_best, y_test)./get_mse(pred0, y_test) .- 1
println("Change to baseline: ", round.(res, digits=5), "\n")



println("Average depth standard: ", average_depth(rf0))
println("Average depth adapted: ", average_depth(rf_best))



n_bins = 20

p1 = plot(d_cv[:α], cv1.bias_list, title="Bias", legend=false)
p2 = plot(d_cv[:α], cv1.variance_list, title="Variance", legend=false)
p3 = plot(d_cv[:α], cv1.mse_list, title="MSE", legend=false)
p4 = plot(d_cv[:α], seq_res, title="Depth", legend=false)
p5 = histogram(pl0, bins=n_bins, legend=false)
p6 = histogram(pl_best, bins=n_bins, legend=false)

l = @layout [a; b; c; d; e f]
plot(p1, p2, p3, p4, p5, p6, layout = l, size=(600, 650))


i = 8
a = zeros(length(rf_best.trees[i].depth_list), 2)
a[:,1] = rf_best.trees[i].depth_list
a[:,2] = rf_best.trees[i].pl

println(a[a[:,2].!=-2,:])


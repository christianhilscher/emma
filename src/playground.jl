using Pkg
using Statistics
using Random, Distributions
using Plots
using SparseGrids
using BenchmarkTools

include("RFR.jl")
include("cross_val.jl")

function friedman(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1] .* x[:,2]) .+ 20 .* (x[:,3] .- 0.5).^2 .+ 10 .* x[:,4] + 5 .* x[:,5] .+ errors

    return Y
end

function summing(x::Matrix, errors::Matrix)

    Y = x[:,1] .+ x[:,2] .+ x[:,3] .+ x[:,4] .+ x[:,5] .+ errors
    return Y
end

function multiplying(x::Matrix, errors::Matrix)

    Y = x[:,1] .* x[:,2] .* x[:,3] .* x[:,4] .* x[:,5] .+ errors
    return Y
end

function summing2(x::Matrix, errors::Matrix)

    Y = x[:,1] .+ x[:,2] .+ x[:,3] .+ x[:,4] .+ x[:,5] .+ cumsum(x[:,6:end]).*0.01 .+ errors
    return Y
end

function make_data(n, d, func)
    x = Array{Float64}(undef, n, d)
    x_train = rand(Uniform(0, 1), n, d)
    x_test = rand(Uniform(0, 1), n, d)
    
    errors_train = rand(n, 1)
    errors_test = zeros(n, 1)

    if func=="friedman"
        y_train = friedman(x_train, errors_train)
        y_test = friedman(x_test, errors_test)
    elseif func=="summing"
        y_train = summing(x_train, errors_train)
        y_test = summing(x_test, errors_test)
    elseif func=="multiplying"
        y_train = multiplying(x_train, errors_train)
        y_test = multiplying(x_test, errors_test)
    elseif func=="summing2"
        y_train = summing2(x_train, errors_train)
        y_test = summing2(x_test, errors_test)
    else
        error("Provide function to compute Y")
    end

    return x_train, x_test, y_train, y_test
end

function get_mse(pred, y)
    bias = mean(pred .- y)
    variance = var(pred)
    mse = mean((pred .- mean(y)).^2)

    return bias, variance, mse
end

Random.seed!(68159)
n = 4000
d = 20

x_train, x_test, y_train, y_test = make_data(n, d, "friedman")

d1 = Dict(:α => collect(LinRange(0, 0.5, 10)),
            :max_features => [5],
            :n_trees => [500])


cv1 = cross_val(d1, random_state = 0)
fit!(cv1, x_train, y_train, nfolds=5)

seq_res = Array{Float64}(undef, length(cv1.regressor_list))
for (ind, rf) in enumerate(cv1.regressor_list)
    seq_res[ind] = strong_selection_freq(rf, 5)
end


println("Correlation between freq. and bias: ", cor(cv1.bias_list, seq_res))
println("Best bias: ", best_model(cv1, "bias").param_dict)
println("Best mse: ", best_model(cv1, "mse").param_dict)


l = @layout [a; b; c; d]
p1 = plot(d1[:α], cv1.bias_list)
p2 = plot(d1[:α], cv1.variance_list)
p3 = plot(d1[:α], cv1.mse_list)
p4 = plot(d1[:α], seq_res)
plot(p1, p2, p3, p4, layout = l)
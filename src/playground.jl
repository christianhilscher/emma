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

    x_train = rand(Uniform(0, 1), n, d)
    x_test = rand(Uniform(0, 1), n, d)
    
    σ = 1
    d = Normal(0, σ)
    td = truncated(d, -Inf, Inf)

    errors_train = rand(td, n, 1)
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
    bias = abs(mean(pred .- y))
    variance = var(pred)
    mse = mean((pred .- mean(y)).^2)

    return bias, variance, mse
end

Random.seed!(68151)
n = 2000
d = 10


times = 10
res_mat = Array{Float64}(undef, (times, 3))

# for i in 1:times

x_train, x_test, y_train, y_test = make_data(n, d, "friedman")

a_list = collect(LinRange(0, 15, 21))

d1 = Dict{Symbol, Vector{Float64}}(
    :max_features => [d],
    :n_trees => [30])

d_cv = copy(d1)
d_cv[:α] = a_list


cv1 = cross_val(d_cv, random_state = 0)
fit!(cv1, x_train, y_train, nfolds=3)

seq_res = Array{Float64}(undef, length(cv1.regressor_list))
for (ind, rf) in enumerate(cv1.regressor_list)
    seq_res[ind] = strong_selection_freq(rf, 5)
end



println("Average depth:, ", average_depth(cv1.regressor_list[4]))
println("Correlation between freq. and bias: ", cor(cv1.bias_list, seq_res))
println("Best bias: ", best_model(cv1, "bias").param_dict)
println("Best mse: ", best_model(cv1, "mse").param_dict)


l = @layout [a; b; c; d]
p1 = plot(d_cv[:α], cv1.bias_list)
p2 = plot(d_cv[:α], cv1.variance_list)
p3 = plot(d_cv[:α], cv1.mse_list)
p4 = plot(d_cv[:α], seq_res)
plot(p1, p2, p3, p4, layout = l)



d0 = copy(d1)
d0[:α] = [0]


d_best = best_model(cv1, "mse").param_dict

rf0 = RFR(param_dict = d0)
rf_best = RFR(param_dict = d_best)

fit!(rf0, x_test, y_test)
fit!(rf_best, x_test, y_test)

pred0 = predict(rf0, x_test)
pred_best = predict(rf_best, x_test)


println("Standard model: ", round.(get_mse(pred0, y_test), digits=5))
println("Adapted model: ", round.(get_mse(pred_best, y_test), digits=5))

res = get_mse(pred_best, y_test)./get_mse(pred0, y_test) .- 1
println("Change to baseline: ", round.(res, digits=5))

    # res_mat[i, :] .= res
# end


average_depth(rf0)
average_depth(rf_best)



pl = combined_splitpoints(rf0)

println("Mean: ", round(mean(pl), digits=5), " and SD: ", round(std(pl), digits=5))

println("Fraction of splits not on median: ", round((sum(pl .!= 0.5)/length(pl)), digits=5))

histogram(pl, bins=20)

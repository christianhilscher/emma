using Pkg
using Statistics
using Random, Distributions
using Distributed
using Plots
using Profile, PProf
include("RFR.jl")
using Base.Threads
using ProgressMeter


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
n = 1000
d = 20

x_train, x_test, y_train, y_test = make_data(n, d, "friedman")



# rf0 = RFR(n_trees = 100, α=0.0, bootstrap=true, random_state=0)
# @time fit!(rf0, x_train, y_train)
# pred_rf0 = predict(rf0, x_test)

# rf05 = RFR(n_trees = 100, α=0.5, bootstrap=true, random_state=0)
# fit!(rf05, x_train, y_train)
# pred_rf05 = predict(rf05, x_test)

# println(get_mse(pred_rf0, y_test))
# println(get_mse(pred_rf05, y_test))






function cv(α_list::Vector{Float64}, x_train::Matrix{Float64}, y_train::Matrix{Float64}, x_test::Matrix{Float64}, y_test::Matrix{Float64})
    res = Array{Float64}(undef, length(α_list), 4)

    @showprogress for (ind, a) in enumerate(α_list)
        rf = RFR(n_trees=100, bootstrap=true, random_state=0, α=a, max_features=20)
        fit!(rf,x_train, y_train)
        pred = predict(rf, x_test)

        res[ind, 1] = a
        res[ind, 2:end] .= get_mse(pred, y_test)
    end

    return res
end

alpha_list = [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 25.0]
test_rest = cv(alpha_list, x_train, y_train, x_test, y_test)

using Pkg
using Statistics
using Random, Distributions
using Distributed
using Plots
using Profile, PProf
include("RFR.jl")
using Base.Threads


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

# Random.seed!(68159)
n = 200
d = 10

x_train, x_test, y_train, y_test = make_data(n, d, "friedman")



rf0 = RFR(n_trees = 50, α=0.0, bootstrap=false, random_state=0)
fit!(rf0, x_train, y_train)
pred_rf0 = predict(rf0, x_test)

rf05 = RFR(n_trees = 50, α=0.5, bootstrap=true, random_state=0)
fit!(rf05, x_train, y_train)
pred_rf05 = predict(rf05, x_test)

println(get_mse(pred_rf0, y_test))
println(get_mse(pred_rf05, y_test))



t = DTRegressor()
m = DTRegressor()
@threads for i in 1:5
    fit!(m, x_train,y_train)
end
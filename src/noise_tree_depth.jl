using Pkg
using Statistics
using Random, Distributions
using Plots
using SparseGrids
using BenchmarkTools
using DataFrames

include("RFR.jl")
include("cross_val.jl")

function friedman(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1] .* x[:,2]) .+ 20 .* (x[:,3] .- 0.5).^2 .+ 10 .* x[:,4] + 5 .* x[:,5] .+ errors

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

    y_train = friedman(x_train, errors_train)
    # y_train = errors_train
    y_test = friedman(x_test, errors_test)

    return x_train, x_test, y_train, y_test
end


# Random.seed!(68151)
n = 5000
d = 5

x_train, x_test, y_train, y_test = make_data(n, d, "friedman")
a_list = collect(LinRange(0, 30, 31))

d1 = Dict{Symbol, Vector{Float64}}(
    :max_features => [d],
    :n_trees => [100],
    :α => [0.0])
    

rf = RFR(param_dict = d1)
fit!(rf, x_train, y_train)


result_arr = Array{Float64}(undef, 0, 2)

for i in 1:length(rf.trees)

    res = zeros(length(rf.trees[i].depth_list), 2) 
    res[:,1] = rf.trees[i].depth_list
    res[:,2] = rf.trees[i].pl

    result_arr = vcat(result_arr, res)
end


dt = DTRegressor(α=0.0)
fit!(dt, x_train, y_train)

result_arr = Array{Float64}(undef, length(dt.depth_list), 2)
result_arr[:,1] = dt.depth_list
result_arr[:,2] = dt.pl

plot_data = result_arr[result_arr[:,2].!=-2, :]
plot_data[:,2] = abs.(plot_data[:,2] .- 0.5)

plot_data = DataFrame(plot_data, :auto)
rename!(plot_data, ["depth", "deviation_from_median"])

gdf = groupby(plot_data, :depth)
cdf = combine(gdf, :deviation_from_median => median)

println(cdf)



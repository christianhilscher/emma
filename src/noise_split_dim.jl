using Plots: layout_args
using Pkg
using Statistics
using Random, Distributions, StatsBase
using Plots
using SparseGrids
using BenchmarkTools
using DataFrames
using StatsPlots

include("RFR.jl")
include("cross_val.jl")

function friedman(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1] .* x[:,2]) .+ 20 .* (x[:,3] .- 0.5).^2 .+ 10 .* x[:,4] + 5 .* x[:,5] .+ errors

    return Y
end

function sine_easy(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1]) .+ errors

    return Y
end

function make_data(n, d, func)

    x_train = rand(Uniform(0, 1), n, d)
    x_test = rand(Uniform(0, 1), n, d)
    
    σ = 8
    d = Normal(0, σ)
    td = truncated(d, -Inf, Inf)

    errors_train = rand(td, n, 1)
    errors_test = zeros(n, 1)

    if func=="friedman"
        y_train = friedman(x_train, errors_train)
        y_test = friedman(x_test, errors_test)
    elseif func=="sine_easy"
        y_train = sine_easy(x_train, errors_train)
        y_test = sine_easy(x_test, errors_test)
    else
        error("Provide function to compute Y")
    end


    return x_train, x_test, y_train, y_test
end

n_runs=15
max_d=50
res_mat = zeros(max_d, n_runs+1)
res_mat[:, 1] = 1:max_d


for r in 2:n_runs+1
    # Random.seed!(68151)
    n = 2000
    d = 10

    x_train, x_test, y_train, y_test = make_data(n, d, "friedman")
    a_list = collect(LinRange(0, 30, 31))

    d1 = Dict{Symbol, Vector{Float64}}(
        :max_features => [d],
        :n_trees => [30],
        :α => [30])


    rf = RFR(param_dict = d1)
    fit!(rf, x_train, y_train)


    result_arr = Array{Float64}(undef, 0, 2)

    for i in 1:length(rf.trees)

        res = zeros(length(rf.trees[i].depth_list), 2) 
        res[:,1] = rf.trees[i].depth_list
        res[:,2] = rf.trees[i].split_dimensions

        result_arr = vcat(result_arr, res)
    end

    plot_data = result_arr[result_arr[:,2].!=-2, :]
    plot_data[:,2] = plot_data[:,2].>5

    plot_data = DataFrame(plot_data, :auto)
    rename!(plot_data, ["depth", "noise"])

    gdf = groupby(plot_data, :depth)
    cdf2 = combine(gdf, :noise => mean)

    res_mat[1:size(cdf2, 1), r] = cdf2[!, :noise_mean] 

end

max_x = 17
y=res_mat[1:max_x,2:end]


plot(1:max_x, y, legend=false, size=(800, 500))
ylims!((0, 1))
title!("Frac. of noisy dimensions = 1/2, error term variance = 8 | Weighted")

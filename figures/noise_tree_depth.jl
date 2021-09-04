# File for generating data for Figure 2

using Pkg
using Random, Distributions
using ProgressMeter
using DataFrames
using Gadfly
using StatsBase

include("/home/christian/UniMA/EMMA/src/RFR.jl")
include("/home/christian/UniMA/EMMA/src/cross_val.jl")
include("/home/christian/UniMA/EMMA/src/aux_functions.jl")

# Friedman function
function friedman(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1] .* x[:,2]) .+ 20 .* (x[:,3] .- 0.5).^2 .+ 10 .* x[:,4] + 5 .* x[:,5] .+ errors

    return Y
end

# Make data function with parameters
function make_data(n, d, func, σ)

    x_train = rand(Uniform(0, 1), n, d)
    x_test = rand(Uniform(0, 1), n, d)
    
    d = Normal(0, σ)
    td = truncated(d, -Inf, Inf)

    errors_train = rand(td, n, 1)
    errors_test = zeros(n, 1)

    if func=="friedman"
        y_train = friedman(x_train, errors_train)
        y_test = friedman(x_test, errors_test)
    else
        error("Provide function to compute Y")
    end


    return x_train, x_test, y_train, y_test
end

# Set seed for reproducability
Random.seed!(68151)

# Set number of runs and allocate space
n_runs = 100
result_arr = Array{Float64}(undef, 0, 2)
σ = 8 # Setting error term variance

@showprogress for run in 1:n_runs

    # Setting number of observations and dimensions
    n = 2000
    d = 5 # Only taking strong variables, we are in subsection: One dimension

    x_train, x_test, y_train, y_test = make_data(n, d, "friedman", σ)
    a_list = collect(LinRange(0, 30, 31))

    # Make parameter dictionary
    d1 = Dict{Symbol, Vector{Float64}}(
        :max_features => [d],
        :n_trees => [100],
        :α => [0.0])


    rf = RFR(param_dict = d1)
    fit!(rf, x_train, y_train)

    for i in 1:length(rf.trees)

        # Save tree depth and P_(t_L)
        res = zeros(length(rf.trees[i].depth_list), 2) 
        res[:,1] = rf.trees[i].depth_list
        res[:,2] = rf.trees[i].pl

        result_arr = vcat(result_arr, res)
    end
end


# Only considering non-terminal nodes
plot_data = result_arr[result_arr[:,2].!=-2, :]
# Calculating λ = 4 * P(t_L) * (1 - P(t_L))
plot_data[:,2] = abs.(plot_data[:,2] .- 0.5)
plot_data[:,2] = 4 .* (plot_data[:,2]) .* (1 .- plot_data[:,2])

# Make dataframe and save results
plot_data = DataFrame(plot_data, :auto)
rename!(plot_data, ["depth", "deviation_from_median"])

jldsave("data/deep_sigma8.jld2"; plot_data)
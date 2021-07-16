using Pkg
using Random, Distributions
using ProgressMeter
using DataFrames
using Gadfly
using StatsBase

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
    elseif func=="sine_easy"
        y_train = sine_easy(x_train, errors_train)
        y_test = sine_easy(x_test, errors_test)
    else
        error("Provide function to compute Y")
    end


    return x_train, x_test, y_train, y_test
end

Random.seed!(68151)

n_runs=11
max_d=50
σ = 1

res_mat = zeros(max_d, n_runs)


@showprogress for r in 1:n_runs
    n = 2000
    d = 10

    x_train, x_test, y_train, y_test = make_data(n, d, "friedman", σ)
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


x_max = 17

res_mat
plot_mat = transpose(res_mat[:,:])
# plot_mat = res_mat
plot_df = DataFrame(plot_mat[:, 1:x_max], :auto)



plot_df

plot_df
plot_df = stack(plot_df)
plot_df[!, "x"] = repeat(1:x_max, inner=n_runs)


plot_df
gdf = groupby(plot_df, :variable)
mean_df = combine(gdf, "value" => mean)


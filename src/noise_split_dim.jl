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
σ = 0.5

res_mat = zeros(max_d, n_runs)


@showprogress for r in 1:n_runs
    n = 3000
    d = 10

    x_train, x_test, y_train, y_test = make_data(n, d, "friedman", σ)
    a_list = collect(LinRange(0, 30, 31))

    d1 = Dict{Symbol, Vector{Float64}}(
        :max_features => [d],
        :n_trees => [30],
        :α => [0.0])


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


x_max = 20

res_mat
wide_df = DataFrame(res_mat[1:x_max, :], :auto)
wide_df_plot = copy(wide_df)
wide_df_plot[!, "depth"] = collect(1:x_max)



mean_df = DataFrame("depth" => collect(1:x_max), "mean" => mean.(eachrow(wide_df)))

p = plot(Coord.cartesian(xmin=0, ymin=0, xmax=x_max, ymax=0.6))
for c in 1:n_runs
    push!(p, layer(wide_df_plot, x=:depth, y=c, Geom.line, alpha=[0.1], Theme(default_color=color("grey"))))
end


push!(p, layer(mean_df, x=:depth, y=:mean, Geom.line, size=[2]))

draw(PNG("graphs/wide_sigma05.png", 20cm, 12cm, dpi=300), p)


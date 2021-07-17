using Pkg
using Random, Distributions
using ProgressMeter
using DataFrames
using Gadfly
using StatsBase

import Cairo, Fontconfig

include("RFR.jl")
include("cross_val.jl")
include("aux_functions.jl")

function get_data(n_runs, max_d, σ)

    res_mat = zeros(max_d, n_runs)


    @showprogress for r in 1:n_runs
        n = 4000
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
    return res_mat
end


function data_to_plot(res_mat::Matrix, x_max::Int, σ::Int)
    wide_df = DataFrame(res_mat[1:x_max, :], :auto)
    wide_df_plot = DataFrame()
    wide_df_plot[!, "depth"] = collect(1:x_max)
    wide_df_plot[!, "mean"] = mean.(eachrow(wide_df))


    wide_df_plot[!, "ymin"] = minimum.(eachrow(wide_df))
    wide_df_plot[!, "ymax"] = maximum.(eachrow(wide_df))
    wide_df_plot[!, "σ"] = repeat([σ], size(wide_df_plot, 1))

    # push!(p, layer(wide_df_plot, x=:depth, y=:mean, ymin=:ymin, ymax=:ymax, Geom.line, Geom.ribbon, alpha=[0.6], color=["Sigma = $σ"]))

    return wide_df_plot

end


Random.seed!(68151)

n_runs=50
max_d=50

# Getting data
res_mat1 = get_data(n_runs, max_d, 1)
res_mat3 = get_data(n_runs, max_d, 3)
res_mat8 = get_data(n_runs, max_d, 8)


r1 = data_to_plot(res_mat1, x_max, 1)
r2 = data_to_plot(res_mat3, x_max, 3)
r3 = data_to_plot(res_mat8, x_max, 8)

plot_df = vcat(r1, r2, r3)

# Plotting
x_max = 20
p = plot(Scale.color_discrete_manual("red", "deepskyblue", "grey"))

push!(p, layer(plot_df, x=:depth, y=:mean, ymin=:ymin, ymax=:ymax, color=:σ, Geom.line, Geom.ribbon, alpha=[0.6]))


push!(p, Guide.YLabel("% of splits on noisy variables"))
push!(p, Guide.XLabel("Tree Depth"))
push!(p, Guide.title("Impact of error term variance on choice of splitting dimension"))

draw(PNG("graphs/wide_sigmas.png", 20cm, 12cm, dpi=300), p)
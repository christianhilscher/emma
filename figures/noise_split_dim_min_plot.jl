## Figure 5

using JLD2
using Statistics
using DataFrames
using Gadfly
using CSV

import Cairo, Fontconfig

### Load data
load_dict = load("data/wide_sigma8.jld2")
res_mat1 = load_dict["out_list"][1]
res_mat1_const = load_dict["out_list"][2]

function data_to_plot(res_mat::Matrix, x_max::Int, σ)
    wide_df = DataFrame(res_mat[1:x_max, :], :auto)
    wide_df_plot = DataFrame()
    wide_df_plot[!, "depth"] = collect(1:x_max)
    wide_df_plot[!, "mean"] = mean.(eachrow(wide_df))


    wide_df_plot[!, "ymin"] = quantile.(eachrow(wide_df), 0.1)
    wide_df_plot[!, "ymax"] = quantile.(eachrow(wide_df), 0.9)
    wide_df_plot[!, "Approach"] = repeat([String(σ)], size(wide_df_plot, 1))

    return wide_df_plot

end

## Transform data
x_max = 20
r1 = data_to_plot(res_mat1, x_max, "weighted")
r2 = data_to_plot(res_mat1_const, x_max, "CART")

plot_df = vcat(r2, r1)

## Plot
p = plot(Scale.color_discrete_manual("#264653", "#E7C15F"))

push!(p, layer(plot_df, x=:depth, y=:mean, ymin=:ymin, ymax=:ymax, color="Approach", Geom.line, Geom.ribbon, alpha=[0.5]))


push!(p, Guide.YLabel("% of splits on noisy variables"))
push!(p, Guide.XLabel("Tree Depth"))
push!(p, Guide.title("σ^2 = 8"))
push!(p, Guide.yticks(ticks=0:0.25:0.8))
# push!(p, Theme(key_position=:none, line_width=0.8mm))
push!(p, Theme(line_width=0.8mm))



draw(PNG("/home/christian/UniMA/EMMA/figures/graphs/wide_sigmas8.png", 20cm, 12cm, dpi=300), p)

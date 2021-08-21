using JLD2
using DataFrames
using Gadfly

tmp_dict = load("data/tree_depth.jld2")
df_tmp = tmp_dict["df_tmp"]


df_plot = stack(df_tmp)
rename!(df_plot, "variable" => "Approach")

p = plot(Scale.color_discrete_manual("#264653", "#E7C15F"), Scale.x_discrete,
Theme(bar_spacing=2mm))

push!(p, layer(df_plot, x=:n, y=:value, color=:Approach, Geom.bar(position=:dodge)))
push!(p, Guide.YLabel("Tree Depth"))
push!(p, Guide.XLabel("Observations n"))

draw(PNG("figures/graphs/tree_depth.png", 20cm, 12cm, dpi=300), p)
## Figure 2

using Pkg
using Gadfly
using JLD2

import Cairo, Fontconfig

# Plotting

load_dict = load("data/noise_split_dim.jld2")
p1 = load_dict["plot_df"]
rename!(p1, "approach" => "σ^2")

p = plot(Scale.color_discrete_manual("#E7C15F", "#C33149", "#2A9D8F"))

push!(p, layer(p1, x=:depth, y=:mean, ymin=:ymin, ymax=:ymax, color="σ^2", Geom.line, Geom.ribbon, alpha=[0.6]))


push!(p, Guide.YLabel("% of splits on noisy variables"))
push!(p, Guide.XLabel("Tree Depth"))
push!(p, Guide.title("Comparison of approaches"))
push!(p, Theme(line_width=0.6mm))

draw(PNG("figures/graphs/wide_sigmas.png", 20cm, 12cm, dpi=300), p)
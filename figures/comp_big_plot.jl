using JLD2
using Statistics
using DataFrames
using Gadfly

import Cairo, Fontconfig

### Load data
load_dict = load("data/comp_big.jld2")
res = load_dict["res"]

gdf = groupby(res, [:n, :Approach])
plot_df = combine(gdf, :bias => mean,
                        :variance => mean,
                        :mse => mean,
                        :bias => (x -> quantile(x, 0.1)) => :bias_min,
                        :variance => (x -> quantile(x, 0.1)) => :variance_min,
                        :mse => (x -> quantile(x, 0.1)) => :mse_min,
                        :bias => (x -> quantile(x, 0.9)) => :bias_max,
                        :variance => (x -> quantile(x, 0.9)) => :variance_max,
                        :mse => (x -> quantile(x, 0.9)) => :mse_max)


p = plot(Scale.color_discrete_manual("#011627", "#ffc000", "#E71D36", "#0F7173"), Scale.x_discrete)
push!(p, layer(plot_df, x=:n, y=:mse_mean, ymin=:mse_min, ymax=:mse_max, color=:Approach, Geom.line, Geom.ribbon, alpha=[0.6]))

push!(p, Guide.YLabel(""))
push!(p, Guide.XLabel("Observations n"))
push!(p, Guide.title("MSE"))

### Output figure
draw(PNG("figures/graphs/comp_big_mse.png", 20cm, 12cm, dpi=300), p)
using JLD2
using Statistics
using DataFrames
using Gadfly

import Cairo, Fontconfig




### Only with 1 repetition
load_dict = load("data/comp_baseline.jld2")
res = load_dict["res"]

p = plot(Scale.color_discrete_manual("grey", "#ffc000"))
push!(p, layer(res, x=:n, y=:bias, color=:Approach, Geom.line))

### Using ribbons, 10 reps
load_dict = load("data/comp_baseline_ribbon.jld2")
res = load_dict["res"]

gdf = groupby(res, [:n, :Approach])
plot_df = combine(gdf, :bias => mean,
                        :variance => mean,
                        :mse => mean,
                        :bias => (x -> quantile(x, 0.1)) => :bias_min,
                        :variance => (x -> quantile(x, 0.1)) => :variance_min,
                        :mse => (x -> quantile(x, 0.1)) => :mean_min,
                        :bias => (x -> quantile(x, 0.9)) => :bias_max,
                        :variance => (x -> quantile(x, 0.9)) => :variance_max,
                        :mse => (x -> quantile(x, 0.9)) => :mean_max)


p = plot(Scale.color_discrete_manual("grey", "#ffc000"))
push!(p, layer(plot_df, x=:n, y=:bias_mean, ymin=:bias_min, ymax=:bias_max, color=:Approach, Geom.line))


push!(p, layer(plot_df, x=:n, y=:bias_mean, ymin=:bias_min, ymax=:bias_max, color=:Approach, Geom.line, Geom.ribbon, alpha=[0.5]))


### Output figure
draw(PNG("figures/graphs/comp_baseline_20_reps.png", 20cm, 12cm, dpi=300), p)
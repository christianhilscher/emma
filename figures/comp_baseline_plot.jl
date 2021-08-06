using JLD2
import Statistics
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
plot_df = load_dict["res"]

gdf = groupby(res, [:n, :Approach])
plot_df = combine(gdf, :bias => mean,
                        :variance => mean,
                        :mse => mean,
                        :bias => minimum,
                        :variance => minimum,
                        :mse => minimum,
                        :bias => maximum,
                        :variance => maximum,
                        :mse => maximum)


plot_df


p = plot(Scale.color_discrete_manual("grey", "#ffc000"))
push!(p, layer(plot_df, x=:n, y=:bias_mean, ymin=:bias_minimum, ymax=:bias_maximum, color=:Approach, Geom.line))


push!(p, layer(plot_df, x=:n, y=:bias_mean, ymin=:bias_minimum, ymax=:bias_maximum, color=:Approach, Geom.line, Geom.ribbon, alpha=[0.5]))


### Output figure
# draw(PNG("figures/graphs/tree_depth.png", 20cm, 12cm, dpi=300), p)
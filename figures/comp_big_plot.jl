using JLD2
using Statistics
using DataFrames
using Gadfly
using CSV

import Cairo, Fontconfig

### Load data
load_dict = load("data/comp_friedman.jld2")
res = load_dict["res"]

gdf = groupby(res, [:n, :Approach])
plot_df = combine(gdf, :bias => mean,
                        :variance => mean,
                        :mse => mean,
                        :bias => (x -> quantile(x, 0.05)) => :bias_min,
                        :variance => (x -> quantile(x, 0.05)) => :variance_min,
                        :mse => (x -> quantile(x, 0.05)) => :mse_min,
                        :bias => (x -> quantile(x, 0.95)) => :bias_max,
                        :variance => (x -> quantile(x, 0.95)) => :variance_max,
                        :mse => (x -> quantile(x, 0.95)) => :mse_max)


p = plot(Scale.color_discrete_manual("#011627", "#ffc000", "#E71D36", "#0F7173"),Guide.xticks(ticks=[0, 5000, 10000, 15000]))
push!(p, layer(plot_df, x=:n, y=:variance_mean, ymin=:variance_min, ymax=:variance_max, color=:Approach, Geom.line, Geom.ribbon, alpha=[0.6]))

push!(p, Guide.YLabel(""))
push!(p, Guide.XLabel("Observations n"))
push!(p, Guide.title("Variance"))

push!(p, Theme(key_position=:none))

### Output figure
draw(PNG("figures/graphs/comp/friedman_variance.png", 20cm, 12cm, dpi=300), p)

### Make table

function comp_table(df::DataFrame)
    
    # Selecting those values with CART Approach as baseline
    base_df = df[df.Approach .== "CART", :]

    rename!(base_df, :bias_mean => :bias_base,
                        :variance_mean => :variance_base,
                        :mse_mean => :mse_base)

    # Merging onto original frame
    out_df = leftjoin(df, base_df[!, [:n, :bias_base, :variance_base, :mse_base]], on=:n)

    # Getting percentages
    out_df[!, :bias_diff] = round.(out_df[!, :bias_mean] ./ out_df[!, :bias_base] .- 1, digits = 3)
    out_df[!, :variance_diff] = round.(out_df[!, :variance_mean] ./ out_df[!, :variance_base] .- 1, digits=3)
    out_df[!, :mse_diff] = round.(out_df[!, :mse_mean] ./ out_df[!, :mse_base] .- 1, digits = 3)
    
    return out_df[!, [:n, :Approach, :bias_diff, :variance_diff, :mse_diff]]
end



tmp = comp_table(plot_df)

variable = :mse_diff
out_table = unstack(tmp[!, [:n, :Approach, variable]], :Approach, variable)

CSV.write("figures/graphs/comp/friedman_mse.csv", out_table)
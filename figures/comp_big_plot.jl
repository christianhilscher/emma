# File for writing comparison table results

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




function make_plot(df::DataFrame, variable::String)

    y_mean = Symbol(string(variable, "_mean"))
    y_min = Symbol(string(variable, "_min"))
    y_max = Symbol(string(variable, "_max"))


    p = plot(Scale.color_discrete_manual("#264653", "#E7C15F", "#2A9D8F", "#C33149"),Guide.xticks(ticks=[0, 5000, 10000, 15000]))
    push!(p, layer(df, x=:n, y=y_mean, ymin=y_min, ymax=y_max, color=:Approach, Geom.line, Geom.ribbon, alpha=[0.6]))

    push!(p, Guide.YLabel(""))
    push!(p, Guide.XLabel("Observations n"))

    if variable!="mse"
        push!(p, Theme(key_position=:none, line_width=0.6mm))
        push!(p, Guide.title(uppercasefirst(variable)))
    else
        push!(p, Guide.title(uppercase(variable)))
        push!(p, Theme(line_width=0.6mm))
    end

    return p
end

p1 = make_plot(plot_df, "bias")
p2 = make_plot(plot_df, "variance")
p3 = make_plot(plot_df, "mse")


### Output figure
draw(PNG("figures/graphs/comp/friedman_bias.png", 20cm, 12cm, dpi=300), p1)
draw(PNG("figures/graphs/comp/friedman_variance.png", 20cm, 12cm, dpi=300), p2)
draw(PNG("figures/graphs/comp/friedman_mse.png", 22cm, 12cm, dpi=300), p3)


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
    out_df[!, :bias_diff] = round.((out_df[!, :bias_mean] ./ out_df[!, :bias_base] .- 1) .* 100, digits = 1)
    out_df[!, :variance_diff] = round.((out_df[!, :variance_mean] ./ out_df[!, :variance_base] .- 1) .* 100, digits=1)
    out_df[!, :mse_diff] = round.((out_df[!, :mse_mean] ./ out_df[!, :mse_base] .- 1) .* 100, digits = 1)
    
    return out_df[!, [:n, :Approach, :bias_diff, :variance_diff, :mse_diff]]
end



tmp = comp_table(plot_df)

variable = :mse_diff
out_table = unstack(tmp[!, [:n, :Approach, variable]], :Approach, variable)
println(out_table)

perc = string.(out_table[!, ["weighted", "max depth", "min samples leaf"]], "%")
println(hcat(out_table[!, [:n, :CART]], perc))
CSV.write("figures/graphs/comp/friedman_variance.csv", hcat(out_table[!, [:n, :CART]], perc))
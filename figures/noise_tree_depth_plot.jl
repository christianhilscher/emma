## Figure 1

using JLD2
using Statistics, Distributions
using StatsBase
using DataFrames
using Gadfly
using CSV

import Cairo, Fontconfig

### Load data
load_dict = load("data/deep_sigma1.jld2")
plot_data = load_dict["plot_data"]


tmp = plot_data[plot_data.depth .< 5, :]

function EmpiricalDistribution(data::Vector{T} where T <: Real)
    sort!(data) #sort the observations
    empirical_cdf = ecdf(data) #create empirical cdf
    data_clean = unique(data) #remove duplicates to avoid allunique error
    cdf_data = empirical_cdf.(data_clean) #apply ecdf to data
    pmf_data = vcat(cdf_data[1],diff(cdf_data)) #create pmf from the cdf
    DiscreteNonParametric(data_clean,pmf_data) #define distribution
end




a1 = EmpiricalDistribution(tmp[tmp.depth.==1,:deviation_from_median])
a2 = EmpiricalDistribution(tmp[tmp.depth.==2,:deviation_from_median])
a3 = EmpiricalDistribution(tmp[tmp.depth.==3,:deviation_from_median])
a4 = EmpiricalDistribution(tmp[tmp.depth.==4,:deviation_from_median])


colors = ["#264653", "#E7C15F", "#2A9D8F", "#C33149"]
depths = ["1", "2", "3", "4"]
dist = [a1, a2, a3, a4]



p = plot(Guide.xticks(ticks=collect(0:0.2:1)), Guide.yticks(ticks=collect(0:0.2:1)));

for i in 1:length(dist)
    push!(p, layer(x=dist[i].support, y=cumsum(dist[i].p), Geom.line, Theme(default_color=color(colors[i]), line_width=0.8mm)));
end

push!(p, Guide.manual_color_key("Depth", depths, colors))
push!(p, Guide.title("Error term variance = 1"))
push!(p, Guide.YLabel("CFD"))
push!(p, Guide.XLabel("λ"))
# push!(p, Theme(line_width=1cm))
# push!(p, Theme(key_position=:none))

draw(PNG("figures/graphs/deep_sigma1.png", 20cm, 12cm, dpi=300), p)
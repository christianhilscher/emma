using Pkg
using Random, Distributions
using ProgressMeter
using DataFrames
using Gadfly
using StatsBase

include("RFR.jl")
include("cross_val.jl")

function friedman(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1] .* x[:,2]) .+ 20 .* (x[:,3] .- 0.5).^2 .+ 10 .* x[:,4] + 5 .* x[:,5] .+ errors

    return Y
end

function sine_easy(x::Matrix, errors::Matrix)
    
    Y = 10 .* sin.(π .* x[:,1]) .+ errors

    return Y
end

function make_data(n, d, func, σ)

    x_train = rand(Uniform(0, 1), n, d)
    x_test = rand(Uniform(0, 1), n, d)
    
    d = Normal(0, σ)
    td = truncated(d, -Inf, Inf)

    errors_train = rand(td, n, 1)
    errors_test = zeros(n, 1)

    if func=="friedman"
        y_train = friedman(x_train, errors_train)
        y_test = friedman(x_test, errors_test)
    elseif func=="sine_easy"
        y_train = sine_easy(x_train, errors_train)
        y_test = sine_easy(x_test, errors_test)
    else
        error("Provide function to compute Y")
    end


    return x_train, x_test, y_train, y_test
end


Random.seed!(68151)

n_runs = 80
result_arr = Array{Float64}(undef, 0, 2)
σ = 8

@showprogress for run in 1:n_runs


    n = 2000
    d = 5

    x_train, x_test, y_train, y_test = make_data(n, d, "friedman", σ)
    a_list = collect(LinRange(0, 30, 31))

    d1 = Dict{Symbol, Vector{Float64}}(
        :max_features => [d],
        :n_trees => [100],
        :α => [0.0])


    rf = RFR(param_dict = d1)
    fit!(rf, x_train, y_train)

    for i in 1:length(rf.trees)

        res = zeros(length(rf.trees[i].depth_list), 2) 
        res[:,1] = rf.trees[i].depth_list
        res[:,2] = rf.trees[i].pl

        result_arr = vcat(result_arr, res)
    end
end

result_arr

plot_data = result_arr[result_arr[:,2].!=-2, :]
plot_data[:,2] = abs.(plot_data[:,2] .- 0.5)
plot_data[:,2] = 4 .* (plot_data[:,2]) .* (1 .- plot_data[:,2])

plot_data = DataFrame(plot_data, :auto)
rename!(plot_data, ["depth", "deviation_from_median"])

gdf = groupby(plot_data, :depth)
cdf2 = combine(gdf, :deviation_from_median => mean)

println(cdf2)



ghi = plot_data[plot_data.depth .< 5, :]

function EmpiricalDistribution(data::Vector{T} where T <: Real)
    sort!(data) #sort the observations
    empirical_cdf = ecdf(data) #create empirical cdf
    data_clean = unique(data) #remove duplicates to avoid allunique error
    cdf_data = empirical_cdf.(data_clean) #apply ecdf to data
    pmf_data = vcat(cdf_data[1],diff(cdf_data)) #create pmf from the cdf
    DiscreteNonParametric(data_clean,pmf_data) #define distribution
end




a1 = EmpiricalDistribution(ghi[ghi.depth.==1,:deviation_from_median])
a2 = EmpiricalDistribution(ghi[ghi.depth.==2,:deviation_from_median])
a3 = EmpiricalDistribution(ghi[ghi.depth.==3,:deviation_from_median])
a4 = EmpiricalDistribution(ghi[ghi.depth.==4,:deviation_from_median])


colors = ["deepskyblue", "green", "orange", "purple"]
depths = ["1", "2", "3", "4"]
dist = [a1, a2, a3, a4]
p = plot();

for i in 1:length(dist)
    push!(p, layer(x=dist[i].support, y=cumsum(dist[i].p), Geom.line, Theme(default_color=color(colors[i]))));
end

push!(p, Guide.manual_color_key("Depth", depths, colors))
push!(p, Guide.title("Error term variance = $σ"))
push!(p, Guide.YLabel(nothing))
push!(p, Guide.XLabel("λ"))

draw(PNG("graphs/deep_sigma8.png", 20cm, 12cm, dpi=300), p)

using Pkg
using Random, Distributions
using Statistics
using ProgressMeter
using DataFrames
using JLD2

import Cairo, Fontconfig

include("/home/christian/UniMA/EMMA/src/RFR.jl")
include("/home/christian/UniMA/EMMA/src/cross_val.jl")
include("/home/christian/UniMA/EMMA/src/aux_functions.jl")


d = 10
σ = 1
m_features = d
cv_parameter = collect(1:5:31)


d_base = Dict{Symbol, Vector{Float64}}(
    :max_features => [m_features],
    :n_trees => [30])

d0 = copy(d_base)
d0[:α] = [0.0]

d1 = copy(d_base)
d1[:α] = cv_parameter


n1 = 4000
x_train, x_test, y_train, y_test = make_data(n1, d, "friedman", σ)

cv = cross_val(d1, random_state=0)
fit!(cv, x_train, y_train)

rf_α = best_model(cv, "mse")
rf_0 = RFR(param_dict = d0)
fit!(rf_0, x_train, y_train)


rf_0_lambda, rf_0_depth = 
rf_α_lambda, rf_α_depth = lambda_depth(rf_α)

df0 = DataFrame()
df0[!, "lambda"] = lambda_depth(rf_0)[1]
df0[!, "depth"] = lambda_depth(rf_0)[2]
df0[!, "Approach"] = repeat([String("Unweighted")], size(df0, 1))

df1 = DataFrame()
df1[!, "lambda"] = lambda_depth(rf_α)[1]
df1[!, "depth"] = lambda_depth(rf_α)[2]
df1[!, "Approach"] = repeat([String("Weighted")], size(df1, 1))

df2 = append!(df0, df1)
df2[!, "lambda"] = 4 .* df2[!, "lambda"] .* (1 .- df2[!, "lambda"])

df3 = filter(row -> row.depth < 20, df2)

gdf = groupby(df3, [:depth, :Approach])
plot_df = combine(gdf, :lambda => mean,
                        :lambda => (x -> quantile(x, 0.1)) => :lambda_min,
                        :lambda => (x -> quantile(x, 0.9)) => :lambda_max)

plot_df

using Gadfly

p = plot(Scale.color_discrete_manual("grey", "#ffc000"))
push!(p, layer(plot_df, x=:depth, y=:lambda_mean, ymin=:lambda_min, ymax=:lambda_max, color=:Approach, Geom.line, Geom.ribbon, alpha=[0.5]))
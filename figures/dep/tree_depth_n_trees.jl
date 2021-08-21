using Pkg
using Statistics
using Random, Distributions
using SparseGrids
using BenchmarkTools
using DataFrames
using Gadfly

include("/home/christian/UniMA/EMMA/src/RFR.jl")
include("/home/christian/UniMA/EMMA/src/cross_val.jl")
include("/home/christian/UniMA/EMMA/src/aux_functions.jl")

import Cairo, Fontconfig

Random.seed!(68151)
n = 2000
d = 10
σ = 1
m_features = d

d1 = Dict{Symbol, Vector{Float64}}(
    :max_features => [m_features],
    :n_trees => [30],
    :α => [2])

d2 = copy(d1)
d2[:α] = [0]

x_train, x_test, y_train, y_test = make_data(n, d, "friedman", σ)

n_list = [5, 10, 30, 50, 100, 200]
res_mat = zeros(length(n_list), 2)


for (ind,n1) in enumerate(n_list)
    
    d1[:n_trees] = n1
    d2[:n_trees] = n1
    
    cv = cross_val(d1, random_state=0)
    fit!(cv, x_train, y_train)

    rf_α = best_model(cv, "mse")
    rf0 = RFR(param_dict = d2)
    fit!(rf0, x_test, y_test)

    pred_α = predict(rf_α, x_test)
    pred0 = predict(rf0, x_test)

    res_mat[ind, 1] = average_depth(rf0)
    res_mat[ind, 2] = average_depth(rf_α)
    println(ind)
end

df_tmp = DataFrame(res_mat, :auto)
rename!(df_tmp, ["unweighted", "weighted"])
df_tmp[!, :n] = n_list

df_plot = stack(df_tmp)
rename!(df_plot, "variable" => "Approach")

p = plot(Scale.color_discrete_manual("grey", "#ffc000"), Scale.x_discrete,
Theme(bar_spacing=2mm))

push!(p, layer(df_plot, x=:n, y=:value, color=:Approach, Geom.bar(position=:dodge)))
push!(p, Guide.YLabel("Tree Depth"))

draw(PNG("figures/graphs/tree_depth.png", 20cm, 12cm, dpi=300), p)
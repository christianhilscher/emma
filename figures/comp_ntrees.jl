using Base: color_normal
using Pkg
using Random, Distributions
using Statistics
using ProgressMeter
using DataFrames
using Gadfly

import Cairo, Fontconfig

include("/home/christian/UniMA/EMMA/src/RFR.jl")
include("/home/christian/UniMA/EMMA/src/cross_val.jl")
include("/home/christian/UniMA/EMMA/src/aux_functions.jl")


Random.seed!(68151)
n = 1000
d = 10
σ = 1
m_features = d



d_base = Dict{Symbol, Vector{Float64}}(
    :max_features => [m_features],
    :n_trees => [30])

d0 = copy(d_base)
d0[:α] = [0.0]

d1 = copy(d_base)
d1[:α] = collect(LinRange(0, 30, 16))

d2 = copy(d_base)
d2[:max_depth] = floor.(collect(LinRange(3, 30, 16)))

d3 = copy(d_base)
d3[:min_samples_leaf] = collect(LinRange(0, 30, 16))

n_list = collect(100:100:1000)
res_mat = zeros(length(n_list), 12)

@showprogress for (n_ind, n1) in enumerate(n_list)

    for d in [d0, d1, d2, d3]
        d[:n_trees] = [n1]
    end

    cv0 = cross_val(d0, random_state=0)
    cv1 = cross_val(d1, random_state=0)
    cv2 = cross_val(d2, random_state=0)
    cv3 = cross_val(d3, random_state=0)

    x_train, x_test, y_train, y_test = make_data(n, d, "friedman", σ)
    for (ind, cv) in enumerate([cv0, cv1, cv2, cv3])
        fit!(cv, x_train, y_train)
        rf = best_model(cv, "mse")
        pred = predict(rf, x_test)

        res_mat[n_ind, (((ind-1)*3)+1):(ind*3)] .= get_mse(pred, y_test)

    end
end

df_tmp = DataFrame(res_mat[:, collect(3:3:12)], :auto)
rename!(df_tmp, ["CART", "α", "min samples leaf", "tree depth"])
df_tmp[!,:n_trees] = n_list

df_plot = stack(df_tmp)
rename!(df_plot, "variable" => "Approach")

p = plot(Scale.color_discrete_manual("grey", "#ffc000", "deepskyblue", "red"))
push!(p, layer(df_plot, x=:n_trees, y=:value, color=:Approach, Geom.line))
push!(p, Guide.YLabel("MSE"))

draw(PNG("figures/graphs/comp_ntrees.png", 20cm, 12cm, dpi=300), p)
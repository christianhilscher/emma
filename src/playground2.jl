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
n = 200
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
d2[:max_depth] = floor.(collect(LinRange(5, 30, 16)))

d3 = copy(d_base)
d3[:min_samples_leaf] = collect(LinRange(0, 30, 16))

cv0 = cross_val(d0, random_state=0)
cv1 = cross_val(d1, random_state=0)
cv2 = cross_val(d2, random_state=0)
cv3 = cross_val(d3, random_state=0)

n_list = [200, 500, 1000, 1500, 2000, 2500, 3000, 3500]
res_mat = zeros(length(n_list), 12)

@showprogress for (n_ind, n1) in enumerate(n_list)
    x_train, x_test, y_train, y_test = make_data(n1, d, "friedman", σ)

    for (ind, cv) in enumerate([cv0, cv1, cv2, cv3])
        fit!(cv, x_train, y_train)
        rf = best_model(cv, "mse")
        pred = predict(rf, x_test)

        res_mat[n_ind, (((ind-1)*3)+1):(ind*3)] .= get_mse(pred, y_test)

    end
end


res_mat[:, collect(3:3:12)]


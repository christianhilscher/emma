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

d_α = copy(d_base)
d_α[:α] = [30.0]


n_list = collect(100:100:200)
res = DataFrame()

@showprogress for (n_ind, n1) in enumerate(n_list)
    x_train, x_test, y_train, y_test = make_data(n1, d, "friedman", σ)

        rf0 = RFR(param_dict=d_base)
        rf_α = RFR(param_dict=d_α)

        approaches = ["unweighted", "weighted"]

        for (ind, rf) in enumerate([rf0, rf_α])
            fit!(rf, x_train, y_train)
            pred = predict(rf, x_test)
            mse_arr = get_mse(pred, y_test)

            res = vcat(res, DataFrame(bias=mse_arr[1], 
                                        variance=mse_arr[2],
                                        mse=mse_arr[3],
                                        Approach=approaches[ind],
                                        n=n1))
        end


        # fit!(rf0, x_train, y_train)
        # fit!(rf_α, x_train, y_train)

        # pred0 = predict(rf0, x_test)
        # pred_α = predict(rf_α, x_test)

        
        # res_mat[n_ind, 1:3] .= get_mse(pred0, y_test)
        # res_mat[n_ind, 4:end] .= get_mse(pred_α, y_test)

end

df_tmp = DataFrame(res_mat, :auto)
rename!(df_tmp)
df_tmp[!, :n] =n_list


p = plot()
push!(p, layer(df_tmp, x=:n, y=:x1, Geom.line))
push!(p, layer(df_tmp, x=:n, y=:x4, Geom.line))
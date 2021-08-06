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

function baseline_comp(d_base, d_α, n_list, reps, res=nothing)

    if res===nothing
        # Initiate output DataFrame
        res = DataFrame()
    end

    # Start at index 2 because first is 0
    @showprogress for n1 in n_list[2:end]

        # General variables
        rf0 = RFR(param_dict=d_base)
        rf_α = RFR(param_dict=d_α)

        approaches = ["unweighted", "weighted"]

        # For each repitition do following
        for r in 1:reps
            x_train, x_test, y_train, y_test = make_data(n1, d, "friedman", σ)

            # Loop through approaches
            for (ind, rf) in enumerate([rf0, rf_α])
                # Loop through repetitions
                    fit!(rf, x_train, y_train)
                    pred = predict(rf, x_test)
                    mse_arr = get_mse(pred, y_test)

                    res = vcat(res, DataFrame(bias=mse_arr[1], 
                                                variance=mse_arr[2],
                                                mse=mse_arr[3],
                                                Approach=approaches[ind],
                                                n=n1))
            end
        end
    end

    return res
end

d = 10
σ = 1
m_features = d


d_base = Dict{Symbol, Vector{Float64}}(
    :max_features => [m_features],
    :n_trees => [30])

d_α = copy(d_base)
d_α[:α] = [30.0]


n_list = collect(0:250:10000)
reps = 10

# Load old dataframe
load_dict = load("data/comp_baseline_ribbon.jld2")
res_old = load_dict["res"]

res = baseline_comp(d_base, d_α, n_list, reps, res_old)


jldsave("data/comp_baseline_ribbon.jld2"; res)



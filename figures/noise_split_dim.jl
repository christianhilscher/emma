# File for generating data for Figure 3

using Pkg
using Random, Distributions
using Statistics
using ProgressMeter
using DataFrames

include("/home/christian/UniMA/EMMA/src/RFR.jl")
include("/home/christian/UniMA/EMMA/src/cross_val.jl")
include("/home/christian/UniMA/EMMA/src/aux_functions.jl")


function get_best_model(n::Int, d::Int, type::String, σ::Int, m_features::Int)
    
    xtrain1, xtest1, ytrain1, ytest1 = make_data(n, d, type, σ)

    a_list = collect(LinRange(0, 50, 15))

    d1 = Dict{Symbol, Vector{Float64}}(
        :max_features => [m_features],
        :n_trees => [30],
        :α => a_list)

    cv = cross_val(d1, random_state = 0)
    fit!(cv, xtrain1, ytrain1, nfolds=3)

    return(best_model(cv, "mse"))
end


function get_data(n_runs, max_d, σ, opt_α=false)

    # Allocate space for results
    res_mat = zeros(max_d, n_runs)

    # Setting parameters
    n = 4000
    d = 10
    m_features = d

    if opt_α
        # Getting the best model given error term variance
        rf_opt = get_best_model(n, d, "friedman", σ, m_features)
        println(rf_opt.α)
    end

    @showprogress for r in 1:n_runs

        x_train, x_test, y_train, y_test = make_data(n, d, "friedman", σ)

        if opt_α
            # Fitting the forest with optimally chosen α with new data
            rf = rf_opt
        else
            # Else choose CART
            d1 = Dict{Symbol, Vector{Float64}}(
                :max_features => [m_features],
                :n_trees => [30],
                :α => [0.0])
            rf = RFR(param_dict = d1)
        end


        fit!(rf, x_train, y_train)

        result_arr = Array{Float64}(undef, 0, 2)

        for i in 1:length(rf.trees)

            # Save depth and splitting dimension
            res = zeros(length(rf.trees[i].depth_list), 2) 
            res[:,1] = rf.trees[i].depth_list
            res[:,2] = rf.trees[i].split_dimensions

            result_arr = vcat(result_arr, res)
        end

        # Only comparing non-terminal nodes
        plot_data = result_arr[result_arr[:,2].!=-2, :]
        # Noisy dimensions are all regressors above 5
        plot_data[:,2] = plot_data[:,2].>5

        # Save figures as dataframe
        plot_data = DataFrame(plot_data, :auto)
        rename!(plot_data, ["depth", "noise"])

        # Grouping them by depth
        gdf = groupby(plot_data, :depth)
        # Getting mean of fraction split on noise
        cdf2 = combine(gdf, :noise => mean)

        res_mat[1:size(cdf2, 1), r] = cdf2[!, :noise_mean] 
    end
    return res_mat
end


function data_to_plot(res_mat::Matrix, x_max::Int, σ)
    wide_df = DataFrame(res_mat[1:x_max, :], :auto)
    wide_df_plot = DataFrame()
    wide_df_plot[!, "depth"] = collect(1:x_max)
    wide_df_plot[!, "mean"] = mean.(eachrow(wide_df))

    # Getting minimum and maximum value of fraction in addition to mean
    wide_df_plot[!, "ymin"] = minimum.(eachrow(wide_df))
    wide_df_plot[!, "ymax"] = maximum.(eachrow(wide_df))
    wide_df_plot[!, "approach"] = repeat([String(σ)], size(wide_df_plot, 1))

    return wide_df_plot

end


# Set seed for reproducability
Random.seed!(68151)

n_runs=100
max_d=50 #Max depth allowed - need to set this parameter; max acieved is around 30

# Getting data
res_mat1 = get_data(n_runs, max_d, 1, false)
res_mat3 = get_data(n_runs, max_d, 3, false)
res_mat8 = get_data(n_runs, max_d, 8, false)

# Figure plots until depth of 20
x_max = 20
r1 = data_to_plot(res_mat1, x_max, "1")
r2 = data_to_plot(res_mat3, x_max, "3")
r3 = data_to_plot(res_mat8, x_max, "8")

plot_df = vcat(r1, r2, r3)
jldsave("data/noise_split_dim.jld2"; plot_df)
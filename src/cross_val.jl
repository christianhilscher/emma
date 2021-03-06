# File holding the cross validation object

using Base.Threads
using ProgressMeter
using SparseGrids
include("RFR.jl")



check_random_state(seed::Int) = MersenneTwister(seed)
check_random_state(rng::AbstractRNG) = rng

# Cross Validation object
mutable struct cross_val

    # internal parameters
    mse_list::Vector{Float64}
    bias_list::Vector{Float64}
    variance_list::Vector{Float64}
    regressor_list::Vector{AbstractRegressor}
    dictionary_list::Vector{Dict}
    n_models::Union{Nothing, Int}

    # external parameters
    parameter_list::Dict
    random_state::Union{AbstractRNG, Int}

    # Initializing new object
    cross_val(parameter_list; 
            random_state = 0) = new(
        [], [], [], [], 
        make_single_dicts(parameter_list), nothing,
        parameter_list,
        check_random_state(random_state)
    )
end

# Function for making a single dictionary out of multiple parameters
# Needed when supplying multiple vectors for tuning parameters 
# For example optimizing over α and max_depth
function make_single_dicts(d::Dict)

    names = collect(keys(d))
    # Get all combinations of values from the dictionary
    combs = combvec([float(v) for (k,v) in d])


    result_array = Array{Dict}(undef, length(combs))
    @inbounds for i in 1:length(combs)
        result_array[i] = Dict(zip(names, combs[i]))
    end

    return result_array
end

function initiate_RF!(cv::cross_val)
    cv.n_models = length(cv.dictionary_list)
    # println("Making ", cv.n_models, " different specifications")

    # Allocate space holding the results for each random forest
    cv.regressor_list = Vector{AbstractRegressor}(undef, cv.n_models)

    @inbounds for i in 1:cv.n_models
        rf = RFR(param_dict = cv.dictionary_list[i])
        cv.regressor_list[i] = rf
    end

    # Make room for validation results
    cv.mse_list = Float64[]
    cv.bias_list = Float64[]
    cv.variance_list = Float64[]

    return cv
end

# Function for getting the MSE
function get_mse(pred, y)
    bias = mean(pred .- y)
    variance = var(pred)
    mse = mean((pred .- mean(y)).^2)

    # Simlpe test to see if everything worked
    @assert mse == variance + bias^2 "mse != variance + bias^2"

    return bias, variance, mse
end

# Fitting the cross-validation object to data
function fit!(cv::cross_val, X::Matrix, Y::Matrix; nfolds::Int=3)
    
    # Initializing cross-validation object with random forests
    initiate_RF!(cv)

    incr = Int(floor(size(X, 1)/nfolds))
    println("Fitting ", nfolds, " folds")

    # Iterating over the list of dictionaries to fit the random forests
    for forest in cv.regressor_list

        result_matrix = Array{Float64}(undef, nfolds, 3)
        
        # Iterating through the folds
        ind = 1:1
        @inbounds for i = 1:nfolds
            # Increase interval for the next fold
            ind = (ind[end]:i*incr)

            # Splitting into hold-out sample
            train_cond = (1:size(X,1) .< ind[1]) + (1:size(X,1) .> ind[end]) .== 1
            test_cond = ind[1] .< 1:size(X, 1) .< ind[end]

            x_train = X[train_cond, :]
            x_test = X[test_cond, :]
            y_train = Y[train_cond, :]
            y_test = Y[test_cond, :]

            # Fit and predict given hold-out sample
            fit!(forest, x_train, y_train)
            pred = predict(forest, x_test)

            b, v, m = get_mse(pred, y_test)
            result_matrix[i, :] = [b, v, m]
        end

        # Assign values
        bias, variance, mse = mean(result_matrix, dims=2)
        validation_results!(cv, bias, variance, mse)
    end
end


# Add results to list which holds all results
function validation_results!(cv::cross_val, bias::Float64, variance::Float64, mse::Float64)
    push!(cv.bias_list, bias)
    push!(cv.variance_list, variance)
    push!(cv.mse_list, mse)
end

# Get best model out of all tried for given measure
function best_model(cv::cross_val, measure::String)

    ind = -1
    if measure == "mse"
        ind = cv.mse_list .== minimum(cv.mse_list)
    elseif measure == "bias"
        ind = cv.bias_list .== minimum(cv.bias_list)
    elseif measure == "variance"
        ind = cv.variance_list .== minimum(cv.variance_list)
    end

    return cv.regressor_list[ind][1]
end

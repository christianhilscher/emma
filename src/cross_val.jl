include("RFR.jl")
using SparseGrids
using ProgressMeter
using Base.Threads



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

    cross_val(parameter_list) = new(
        [], [], [], [], 
        make_single_dicts(parameter_list), nothing,
        parameter_list
    )
end

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

function get_mse(pred, y)
    bias = mean(pred .- y)
    variance = var(pred)
    mse = mean((pred .- mean(y)).^2)

    return bias, variance, mse
end

function fit!(cv::cross_val, X::Matrix, Y::Matrix; split::Float64=0.5)

    # Split data into training and validation set
    probs = rand(Uniform(0, 1), size(X, 1))
    x_train = X[probs .<= split, :]
    y_train = Y[probs .<= split, :]
    x_test = X[probs .> split, :]
    y_test = Y[probs .> split, :]
    
    # Initializing cross-validation object with random forests
    initiate_RF!(cv)

    # Iterating over the list of dictionaries to fit the random forests
    for forest in cv.regressor_list
        # Fit and predict 
        fit!(forest, x_train, y_train)
        pred = predict(forest, x_test)

        # Assign values
        bias, variance, mse = get_mse(pred, y_test)
        validation_results!(cv, bias, variance, mse)
    end

end

function validation_results!(cv::cross_val, bias::Float64, variance::Float64, mse::Float64)
    push!(cv.bias_list, bias)
    push!(cv.variance_list, variance)
    push!(cv.mse_list, mse)
end


function best_model(cv::cross_val, measure::String)

    ind = -1
    if measure == "mse"
        ind = cv.mse_list .== minimum(cv.mse_list)
    elseif measure == "bias"
        ind = cv.bias_list .== minimum(cv.bias_list)
    elseif measure == "variance"
        ind = cv.variance_list .== minimum(cv.variance_list)
    end

    println(cv.dictionary_list[ind])
    return cv.regressor_list[ind]
end

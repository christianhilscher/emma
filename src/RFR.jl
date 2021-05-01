using Random
using Base.Threads


include("DecisionTree.jl")

mutable struct RFR <: AbstractRegressor

    # internal variables
    n_features::Union{Int, Nothing}
    trees::Vector{DTRegressor}

    # external parameters
    n_trees::Int
    max_depth::Union{Int,Nothing}
    max_features::Union{Int, Nothing}
    min_samples_leaf::Int
    # random_state::Union{AbstractRNG, Int}
    random_state::Int
    bootstrap::Bool
    α::Float64
    param_dict::Dict

    RFR(; 
        n_trees=100,
        max_depth=nothing,
        max_features=nothing,
        min_samples_leaf=1,
        random_state=0,
        bootstrap=true,
        α=0.0,
        param_dict=Dict()) = new(
            nothing, [], 
            haskey(param_dict, :n_trees) ? param_dict[:n_trees] : n_trees,
            haskey(param_dict, :max_depth) ? param_dict[:max_depth] : max_depth,
            haskey(param_dict, :max_features) ? param_dict[:max_features] : max_features,
            haskey(param_dict, :min_samples_leaf) ? param_dict[:min_samples_leaf] : min_samples_leaf,
            # check_random_state(random_state),
            haskey(param_dict, :random_state) ? param_dict[:random_state] : random_state,
            haskey(param_dict, :bootstrap) ? param_dict[:bootstrap] : bootstrap,
            haskey(param_dict, :α) ? param_dict[:α] : α,
            param_dict)
end

function fit!(forest::RFR, X::Matrix, Y::Matrix)
    @assert size(Y, 2) == 1 "Y must be 1d"

    # set internal variables
    forest.n_features = size(X, 2)

    # Allocate space
    forest.trees = Array{DTRegressor}(undef, forest.n_trees)
    # make the trees

    @inbounds @threads for i in 1:forest.n_trees
        # rng_states[i] = copy(forest.random_state)
        forest.trees[i] = create_tree(forest, X, Y)
    end
    
    return
end

function create_tree(forest::RFR, X::Matrix, Y::Matrix)
    n_samples = size(X, 1)

    # Tree random state is random state + ThreadID
    tree_rs = MersenneTwister(Threads.threadid() + forest.random_state)

    inds = Array{Int}(undef, n_samples)
    if forest.bootstrap
        inds = [rand(tree_rs, 1:n_samples) for i in 1:n_samples]
        unique!(inds)
        X_ = copy(X[inds, :])
        Y_ = copy(Y[inds, :])
    else
        X_ = copy(X)
        Y_ = copy(Y)
    end

    new_tree = DTRegressor(max_depth= forest.max_depth,
                            max_features = forest.max_features,
                            min_samples_leaf = forest.min_samples_leaf,
                            random_state = tree_rs,
                            α = forest.α)
    # This function calls the DTRegressor version of fit!
    fit!(new_tree, X_, Y_)

    return new_tree
end

function predict(forest::RFR, X::Matrix)
    @assert forest.n_features == size(X, 2) "# of features are not the same"

    # Throw error if tree is not fitted
    if length(forest.trees) == 0
        error("Forest is not fitted")
    end


    prediction = Array{Float64}(undef, (size(X, 1), forest.n_trees))

    for (ind, t) in enumerate(forest.trees)
        prediction[:,ind] = predict(t, X)
    end

    out_arr = mean(prediction, dims=2)
    return out_arr
end

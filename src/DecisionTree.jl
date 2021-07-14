check_random_state(seed::Int) = MersenneTwister(seed)
check_random_state(rng::AbstractRNG) = rng

mutable struct BinaryTree
    children_left::Vector{Int}
    children_right::Vector{Int}
    BinaryTree() = new([], [])
end

function add_node!(tree::BinaryTree)
    push!(tree.children_left, -1)
    push!(tree.children_right, -1)
    return
end

function set_left_child!(tree::BinaryTree, node_id::Int64, child_id::Int)
    tree.children_left[node_id] = child_id
    return 
end

function set_right_child!(tree::BinaryTree, node_id::Int64, child_id::Int)
    tree.children_right[node_id] = child_id
    return 
end

function get_children(tree::BinaryTree, node_id::Int64)
    return tree.children_left[node_id], tree.children_right[node_id]
end

function is_leaf(tree::BinaryTree, node_id::Int64)
    return tree.children_left[node_id] == tree.children_right[node_id] == -1
end

###############################################################################
abstract type AbstractRegressor end

mutable struct DTRegressor <: AbstractRegressor

    
    # Internal parameters
    num_nodes::Int
    binarytree::BinaryTree
    n_samples::Vector{Int} # Number of samples for each node
    split_dimensions::Vector{Union{Int, Nothing}} # Which variable was chosen for splitting
    split_values::Vector{Union{Float64, Nothing}} # Split-point values
    yhat::Vector{Union{Float64, Nothing}} # Predictor in each node
    mse::Vector{Union{Float64, Nothing}} # Reduction in MSE when splitting this node
    n_features::Union{Int, Nothing}
    pl::Vector{Float64} # Probability of falling into left node
    pr::Vector{Float64} # Probability of falling into right node
    depth_list::Vector{Union{Int, Nothing}} # Records depth for each split

    # externally set parameters
    max_depth::Union{Int, Nothing}
    max_features::Union{Int, Nothing}
    min_samples_leaf::Int
    random_state::Union{AbstractRNG, Int}
    α::Float64

    DTRegressor(;
        max_depth=nothing,
        max_features=nothing,
        min_samples_leaf = 1,
        random_state=Random.GLOBAL_RNG,
        α = 0.01,)  = new(
            0, BinaryTree(), [], [], [], [], [], nothing, [], [], [],
        max_depth, 
        max_features, 
        min_samples_leaf, 
        check_random_state(random_state), 
        α)

end

function fit!(tree::DTRegressor, X::Matrix, Y::Matrix)
    @assert size(Y, 1) == size(X, 1) "X and Y have different lengths"
    @assert size(Y, 2) == 1 "Y is not a 1d matrix"
    # Set the internal value
    tree.n_features = size(X, 2)
    tree.max_features = isnothing(tree.max_features) ? size(X, 2) : tree.max_features


    # Do the fitting
    split_node!(tree, X, Y, 0)
    return
end

function set_defaults!(tree::DTRegressor, Y::Matrix)
    push!(tree.split_dimensions, -2)
    push!(tree.split_values, nothing)
    push!(tree.n_samples, size(Y, 1))
    push!(tree.mse, 0.0)
    # Here we assign the estimator to be the mean of all Y values in a node
    push!(tree.yhat, mean(Y))
    # As default for node balancedness take -2 
    push!(tree.pl, -2)
    push!(tree.pr, -2)
    add_node!(tree.binarytree)
end

function split_node!(tree::DTRegressor, X::Matrix, Y::Matrix, depth::Int)

    tree.num_nodes += 1
    node_id = tree.num_nodes
    set_defaults!(tree, Y)
    push!(tree.depth_list, depth)

    # In case we have max_features choose randomly that many features to consider
    n_feature_splits = isnothing(tree.max_features) ? tree.n_features : min(tree.n_features, tree.max_features)
    features = randperm(tree.random_state, tree.n_features)[1:n_feature_splits]

    max_mse = 0
    @inbounds for j in features
        # println(size(X))
        max_mse = argmax_j(j, tree, X, Y, node_id, max_mse, depth)
    end
    if max_mse == 0
        return #No split was made
    end
    
    # Make children_left
    if isnothing(tree.max_depth) || (depth < tree.max_depth)

        x_split = X[:, tree.split_dimensions[node_id]] # Split dimension
        lhs = x_split .<= tree.split_values[node_id] # All obs going into left node
        rhs = x_split .> tree.split_values[node_id] # All obs going into right node

        # Making a new tree for left node and all ancestors
        set_left_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        split_node!(tree, X[lhs, :], Y[lhs,:], depth + 1)
        # Making a new tree for right node and all ancestors
        set_right_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        split_node!(tree, X[rhs, :], Y[rhs,:], depth + 1)
    end
    return 
end


function argmax_s(arr::Vector{Float64}, tree::DTRegressor, depth::Int)
    
    α = tree.α
    min_samples = tree.min_samples_leaf

    # Making space
    Δ = 0
    ind = 0

    # Length of array then gives probabilities of where element lands
    n = length(arr)
    # Only look within the min_samples interval
    @inbounds for i in min_samples:n-min_samples-1

        Δ_tmp = get_Δ(arr, i, n, α, depth)
        # Only updating if decrease in variance is higher
        if Δ_tmp > Δ
            Δ = Δ_tmp
            ind = i
        end
    end

    # Retruning decrease and index
    return Δ, ind
    # return (4*(ind/n)*(1 - ind/n))^α * Δ, ind
end

function get_Δ(arr::Vector{Float64}, ind::Int64, n::Int64, α::Float64, depth::Int)
    # Calculating mean in left and right node respectively
    Y_left = mean(arr[1:ind])
    Y_rigth = mean(arr[ind+1:end])

    Δ = (ind/n)*(1 - ind/n)*(Y_left - Y_rigth)^2

    # Return weighted splitpoint. If α = 0 then we have the standard case
    if α == 0.0
        return Δ
    else
        return (4*(ind/n)*(1 - ind/n))^(depth^α) * Δ
    end
end

function argmax_j(j::Int, tree::DTRegressor, X::Matrix, Y::Matrix, node_id::Int, max_mse, depth::Int)


    if size(X, 1) > tree.min_samples_leaf
        # Getting dimension j
        x = X[:, j]
        order = sortperm(x, alg=InsertionSort)

        x_sorted, y_sorted = x[order], vec(Y[order])
        tmp_mse, split_index = argmax_s(y_sorted, tree, depth)
        
        # If new mse is bigger than previous value, use this variable instead
        if tmp_mse > max_mse
            max_mse = tmp_mse
            tree.split_dimensions[node_id] = j
            tree.split_values[node_id] = x_sorted[split_index]

            # Asigning the number of nodes in left and right nodes for this split
            tree.pl[node_id] = split_index / length(x)
            tree.pr[node_id] = 1 - split_index / length(x)

        end
        return max_mse
    else
        return 0
    end
end

###############################################################################

function predict_arr(tree::DTRegressor, arr::Vector)

    next_node = 1
    while !is_leaf(tree.binarytree, next_node)
        left, right = get_children(tree.binarytree, next_node)
        next_node = arr[tree.split_dimensions[next_node]] <= tree.split_values[next_node] ? left : right
    end
    return tree.yhat[next_node]
end


function predict(tree::DTRegressor, X::Matrix)
    @assert tree.n_features == size(X, 2) "# of features are not the same"

    # Throw error if tree is not fitted
    if tree.num_nodes == 0
        error("Tree is not fitted")
    end

    prediction = Vector{Float64}(undef, size(X, 1))

    # Transpose matrix for higher efficiency
    X_prime = transpose(X)
    @inbounds for ind in 1:size(X_prime, 2)
        prediction[ind] = predict_arr(tree, X_prime[:,ind])
    end
    return prediction
end

function strong_selection_freq(tree::DTRegressor, var_index::Int)
    res_arr = 0

    tmp = tree.split_dimensions[tree.split_dimensions .!= -2]
    res_arr = sum(tmp .<= var_index)/length(tmp)

    return res_arr
end
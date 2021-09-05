# File holding the base regressor - a single decision tree

check_random_state(seed::Int) = MersenneTwister(seed)
check_random_state(rng::AbstractRNG) = rng

# Binary Tree as most basic form
mutable struct BinaryTree
    children_left::Vector{Int}
    children_right::Vector{Int}
    BinaryTree() = new([], [])
end

# Adding node; default node number is -1; will be changed
function add_node!(tree::BinaryTree)
    push!(tree.children_left, -1)
    push!(tree.children_right, -1)
    return
end

# Add a node to the left side of the parent node
function set_left_child!(tree::BinaryTree, node_id::Int64, child_id::Int)
    tree.children_left[node_id] = child_id
    return 
end

# Add a node to the right side of the parent node
function set_right_child!(tree::BinaryTree, node_id::Int64, child_id::Int)
    tree.children_right[node_id] = child_id
    return 
end

# Get the children nodes of a particular node
function get_children(tree::BinaryTree, node_id::Int64)
    return tree.children_left[node_id], tree.children_right[node_id]
end

# Check whether node is a terminal node (= leaf)
function is_leaf(tree::BinaryTree, node_id::Int64)
    return tree.children_left[node_id] == tree.children_right[node_id] == -1
end

###############################################################################
abstract type AbstractRegressor end

# Decision Tree Regressor object
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

    # Initialize object with default parameters
    DTRegressor(;
        max_depth=nothing,          # Not binding by default
        max_features=nothing,       # Set with parameters; = mtry
        min_samples_leaf = 1,       # ν = 1 -> Standard CART
        random_state=Random.GLOBAL_RNG,
        α = 0.01,)  = new(
            0, BinaryTree(), [], [], [], [], [], nothing, [], [], [],
        max_depth, 
        max_features, 
        min_samples_leaf, 
        check_random_state(random_state), 
        α)

end

# Fit the function to the data
function fit!(tree::DTRegressor, X::Matrix, Y::Matrix)
    # Sanity checks for dimensions
    @assert size(Y, 1) == size(X, 1) "X and Y have different lengths"
    @assert size(Y, 2) == 1 "Y is not a 1d matrix"
    
    
    # Set the internal value
    tree.n_features = size(X, 2)        # Number of features
    # If max_features is not set use all features
    tree.max_features = isnothing(tree.max_features) ? size(X, 2) : tree.max_features


    # Do the fitting - Apply Algorithm 1 starting with whole data set
    split_node!(tree, X, Y, 0)
    return
end

# Function for setting the defaults
function set_defaults!(tree::DTRegressor, Y::Matrix)

    # Default values are usually -2, it's only a placeholder which will be changes later
    # Enables quick checks to see whether everything is alright
    
    push!(tree.split_dimensions, -2)
    push!(tree.split_values, nothing)
    push!(tree.n_samples, size(Y, 1))   # Sample size of node

    # Default reduction in impurity - corresponds to equation (1)
    push!(tree.mse, 0.0)        

    # Here we assign the estimator to be the mean of all Y values in a node
    push!(tree.yhat, mean(Y))

    # As default for node balancedness take -2 
    push!(tree.pl, -2)
    push!(tree.pr, -2)
    add_node!(tree.binarytree)
end

# This is essentially Algorithm 1; it is applied recursively to each node
function split_node!(tree::DTRegressor, X::Matrix, Y::Matrix, depth::Int)

    tree.num_nodes += 1             # Increase number of total nodes within tree
    node_id = tree.num_nodes        # Reference for node is the node_id
    set_defaults!(tree, Y)          # Set default values
    push!(tree.depth_list, depth)   # Add depth of node to external depth_list

    # In case we have max_features choose randomly that many features to consider
    n_feature_splits = isnothing(tree.max_features) ? tree.n_features : min(tree.n_features, tree.max_features)

    features = randperm(tree.random_state, tree.n_features)[1:n_feature_splits]

    # Default decrease is 0
    max_decrease = 0
    # Loop over all features chosen previously
    @inbounds for j in features
        # Get highest decrease over all features j
        max_decrease = argmax_j(j, tree, X, Y, node_id, max_decrease, depth)
    end

    # If no feature returns a positive decrease, terminate node
    if max_decrease == 0
        return #No split was made
    end


    # Make child nodes if max_depth is not reached yet
    if isnothing(tree.max_depth) || (depth < tree.max_depth)


        x_split = X[:, tree.split_dimensions[node_id]] # Split dimension
        lhs = x_split .<= tree.split_values[node_id] # All obs going into left node
        rhs = x_split .> tree.split_values[node_id] # All obs going into right node

        # Assigning left node as a child node t(L) of the current node t
        set_left_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        # Again calling Algorithm 1 on this child node
        split_node!(tree, X[lhs, :], Y[lhs,:], depth + 1)

        # Assigning right node as a child node t(R) of the current node t
        set_right_child!(tree.binarytree, node_id, tree.num_nodes + 1)
        # Again calling Algorithm 1 on this child node
        split_node!(tree, X[rhs, :], Y[rhs,:], depth + 1)
    end
    return 
end

# Function for maximizing decrease in impurity over all features given
function argmax_j(j::Int, tree::DTRegressor, X::Matrix, Y::Matrix, node_id::Int, max_decrease, depth::Int)

    # Only split if min_samples_leaf condition is not binding
    # Times 2 since each of the child nodes needs to contain at least min_samples_leaf observations
    if (size(X, 1) > 2*tree.min_samples_leaf) && (std(Y)!=0)
        
        # Getting dimension j
        x = X[:, j]
        order = zeros(Int, length(x))           # Allocate space for odered variable
        sortperm!(order, x, alg=InsertionSort)  # Order along dimension j

        x_sorted, y_sorted = x[order], vec(Y[order])
        # Maximizing over split points s; 
        tmp_decrease, split_index = argmax_s(y_sorted, tree, depth)
        
        # If new decrease is bigger than previous value, use this variable instead
        if tmp_decrease > max_decrease
            max_decrease = tmp_decrease
            tree.split_dimensions[node_id] = j                  # Record split dimensions
            tree.split_values[node_id] = x_sorted[split_index]  # Record split point

            # Asigning the number of nodes in left and right nodes for this split
            tree.pl[node_id] = split_index / length(x)      # P(t_L) 
            tree.pr[node_id] = 1 - split_index / length(x)  # P(t_R)

        end
        return max_decrease
    else
        return 0
    end
end

# Maximizing over all split points for a given dimension
function argmax_s(arr::Vector{Float64}, tree::DTRegressor, depth::Int)
    
    # Re-assigining parameters for easier use
    α = tree.α
    min_samples = tree.min_samples_leaf

    # Allocating space
    Δ = 0       # This is Δ(s, t) from equation (2); default is 0
    ind = 0     # Index of split point

    # Length of array then gives probabilities of where element lands
    n = length(arr)
    # Only look within the min_samples interval
    @inbounds for i in min_samples:n-min_samples-1

        # Function for calculating eqaution (2)
        Δ_tmp = get_Δ(arr, i, n, α, depth)
        # Only updating if decrease in variance is higher
        if Δ_tmp > Δ
            Δ = Δ_tmp
            ind = i
        end
    end

    # Retruning decrease and index
    return Δ, ind
end

# This is where the splitting rule comes into play
function get_Δ(arr::Vector{Float64}, ind::Int64, n::Int64, α::Float64, depth::Int)
    # Calculating mean in left and right child node respectively
    Y_left = mean(arr[1:ind])
    Y_rigth = mean(arr[ind+1:end])

    # Decrease in impurity is given by P(t_L)*P(t_R) * (Y_L - Y_R)^2
    # The same as equation (2) but rewritten for efficiency
    Δ = (ind/n)*(1 - ind/n)*(Y_left - Y_rigth)^2

    # This is the weighted splitting rule
    # If α = 0 then we have the standard case

    # Chosen function; α=a compared to paper! Corresponds to equation (18)
    α_k = depth^α

    # Return decrease in impurity from weighted splitting rule
    return (4*(ind/n)*(1 - ind/n))^(α_k) * Δ    # Equation (17)
end

###############################################################################
# Functions used once the tree is built

# Predicting the values of one observation
function predict_arr(tree::DTRegressor, arr::Vector)

    # Start with first node and then drop down through the tree
    next_node = 1
    # While node is not terminal node drop down deeper
    while !is_leaf(tree.binarytree, next_node)
        left, right = get_children(tree.binarytree, next_node)
        next_node = arr[tree.split_dimensions[next_node]] <= tree.split_values[next_node] ? left : right
    end

    # Once terminal node is reached, return yhat from that terminal node 
    return tree.yhat[next_node]
end

# Predict values for a whole matrix
function predict(tree::DTRegressor, X::Matrix)
    @assert tree.n_features == size(X, 2) "# of features are not the same"

    # Throw error if tree is not fitted
    if tree.num_nodes == 0
        error("Tree is not fitted")
    end

    # Allocate space for predictions
    prediction = Vector{Float64}(undef, size(X, 1))

    # Transpose matrix for higher efficiency; looping over columns is faster than over rows
    X_prime = transpose(X)

    # Predicting values for each observation of data set
    @inbounds for ind in 1:size(X_prime, 2)
        prediction[ind] = predict_arr(tree, X_prime[:,ind])
    end
    return prediction
end

# Function for getting the selection frequency of strong variables
# Strong variables are all from 1-S; supply S=var_index
# Used for Figures 3 and 5
function strong_selection_freq(tree::DTRegressor, var_index::Int)
    
    # Allocate space
    res_arr = 0

    # Get splitting dimensions of all nodes which are not terminal nodes
    tmp = tree.split_dimensions[tree.split_dimensions .!= -2]
    
    # Selection frequency
    res_arr = sum(tmp .<= var_index)/length(tmp)

    return res_arr
end
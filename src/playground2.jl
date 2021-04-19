using Pkg
using Statistics
using Random, Distributions
using Plots

function argmax_s(arr::Vector{Float64}, α)
    
    min_samples = 1

    # Making space
    Δ = 0
    ind = 0

    # Length of array then gives probabilities of where element lands
    n = length(arr)
    # Only look within the min_samples interval
    @inbounds for i in min_samples:n-min_samples-1

        Δ_tmp = get_Δ(arr, i, n, α)
        # Only updating if decrease in variance is higher
        if Δ_tmp > Δ
            Δ = Δ_tmp
            ind = i
        end
    end

    # Retruning decrease and index
    # return Δ, ind
    return Δ, ind
end

function get_Δ(arr::Vector{Float64}, ind::Int64, n::Int64, α::Float64)
# Calculating mean in left and right node respectively
    Y_left = mean(arr[1:ind])
    Y_rigth = mean(arr[ind+1:end])

    Δ = (ind/n)*(1 - ind/n)*(Y_left - Y_rigth)^2

    # Return weighted splitpoint. If α = 0 then we have the standard case
    return (4*(ind/n)*(1 - ind/n))^α * Δ
    # return Δ
end


x = range(0.0, 1.0, step=0.001)
y = sin.(20*π*x)
y = rand(Normal(0, 15), length(x))
y= collect(0:0.001:1)

plot(x, y)

α_list = 0:0.005:10

res = Array{Float64}(undef, (length(α_list), 2))

for (ind, a) in enumerate(α_list)
    res[ind, :] .= argmax_s(y, a)
end

relativ = res[:, 1] ./ res[1,1]
plot(α_list, relativ)
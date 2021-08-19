using Pkg
using RDatasets
using StatsModels, Statistics, StatsBase

include("cross_val.jl")
include("/home/christian/UniMA/EMMA/src/aux_functions.jl")

# println(RDatasets.datasets("Ecdat"))

df = dataset("Ecdat", "RetSchool")

dropmissing!(df)



train_n= Int(floor(size(df,1)*0.5))
train_indices = sample(collect(1:size(df, 1)),train_n)

df_mat = Matrix(df)

dt = fit(UnitRangeTransform, df_mat, dims=1)
df_mat = StatsBase.transform(dt, df_mat)

x_train = df_mat[train_indices, :]
y_train = reshape(df_mat[train_indices, 1], :, 1)
x_test = df_mat[Not(train_indices), :]
y_test = reshape(df_mat[Not(train_indices), 1], :, 1)



x_rand = rand(train_n, 500)
x_rand_test = rand(size(x_test, 1), 500)


d1 = Dict{Symbol, Vector{Float64}}(
    :max_features => [size(x_train, 2)],
    :n_trees => [30])

cv = cross_val(d1)
rf = RFR(param_dict=d1)
dt = DTRegressor()

y = randn(train_n, 1)
n=20
xt = hcat(x_train[:, :], x_rand[:,1:n])
xtest = hcat(x_test, x_rand_test[:, 1:n])

fit!(rf, x_rand, y)


p = predict(rf, xtest)
println(get_mse(p, y_test))

using Pkg
using RDatasets

include("cross_val.jl")

println(RDatasets.datasets("Ecdat"))

df = dataset("Ecdat", "OFP")
df = dataset("Ecdat", "Kakadu")

println(first(df, 5))
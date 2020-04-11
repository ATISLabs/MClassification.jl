###Importing dependencies
using Pkg
Pkg.activate("./")
using MLJ, Queryverse
using DataFrames
using MClassification
dataset_path = "./datasets/sinthetic/"
dataset = Queryverse.load(dataset_path * "UG_2C_5D.csv", header_exists=false) |> DataFrame
X = dataset[1:end,1:end-1]
y = coerce(dataset[1:end,end], autotype(dataset[1:end,end]))
classifier = MClassification.fit(X[1:150, :], y[1:150], 0.1)

y_p = MClassification.predict(classifier, convert(Array{Float64, 2}, X[151:end, :]))

println(convert(Array{Float64, 1}, first(X)) .* convert(Array{Float64, 1}, first(X)))

X[1,:] .* X[2,:]
vector = [1 2 5]
teste(vector)
function teste(teste::Array{T}) where {T<:Number}
      println(teste)
end

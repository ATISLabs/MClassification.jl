###Importing dependencies
using Pkg
Pkg.activate("./")
using DataFrames, MClassification, MLJ, Queryverse, Plots
@load KNNClassifier

###Loading the dataset
dataset_path = "./datasets/sinthetic/"
dataset = Queryverse.load(dataset_path * "UG_2C_5D.csv", header_exists=false) |> DataFrame

###Setting the data
X = dataset[:,1:end-1]
y = dataset[:,end]

training_set_size = 150
y_predicted = [Array{Any, 1}() for i in 1:2]

###KNN
knn_classifier = MLJ.machine(KNNClassifier(K=3), X, coerce(y, autotype(y)))
train, test = (collect(1:training_set_size), collect(training_set_size+1:length(y)))
MLJ.fit!(knn_classifier, rows=train)
y_predicted[1] = predict_mode(knn_classifier, rows=test)

###MClassification
m_classifier = MClassification.fit(X[1:150, :], y[1:150], 0.1)
y_predicted[2] = MClassification.predict(m_classifier, convert(Array{Float64, 2}, X[151:end, :]))

n_instances = 0
hates = [Array{Float64, 1}() for i in 1:2]
hits = zeros(Int64, 2)
for i in 1:length(y_predicted[2])
    global n_instances += 1

    if y[i+150] == y_predicted[1][i]
        hits[1] += 1
    end

    if y[i+150] == y_predicted[2][i]
        hits[2] += 1
    end

    if n_instances % 2000 == 0
            append!(hates[1], hits[1] / 2000)
            append!(hates[2], hits[2] / 2000)
            hits[1] = 0
            hits[2] = 0
    end
end
if(hits[1] > 0 || hits[2] > 0)
    append!(hates[1], hits[1] / (length(y_predicted[2])%2000))
    append!(hates[2], hits[2] / (length(y_predicted[2])%2000))
end
plot(hates, ylims = (0, 1), labels = permutedims(["Static-KNN", "MClassification"]), legend=:outertopright)

y_test = y[151:end]

accuracy =
if y_test[i]==
y_predicted
function Accuracy(y::Array{T, N}, y_predicted::Array{T, N}, steps=1) where {T == T}
{
    if length(y_predicted) != length(y)
        return catch
    end
    hits = 0
    for i in 1:length(y)
        if y[i] == y_predicted[i]
            hits += 1
        end
    end
}

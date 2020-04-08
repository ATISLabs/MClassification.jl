###Importing dependencies
using Pkg, DataFrames
Pkg.activate("../juliadev/")
include("../src/mclassifier.jl")

###Setting path of the datasets and getting the their names
path = "../datasets/sinthetic/"
datasets = split(read(`ls $path`, String), "\n")

dataset_results = DataFrame()

for dataset in datasets[1:6]
    n_instancias = 0
    hits = 0
    misses = 0

    classifier = MClassifier(0.1)
    classifier.n_features = 2

    dataset_file = open(path * dataset , "r")
    MClassifier_initialize(classifier, dataset_file, 150)

    #while true
    for i in 1:15000
        n_instancias += 1
        new_instance = zeros(Float64, classifier.n_features)
        rel1 = load_instance(dataset_file, 2, new_instance)
        if rel1 == ""
            break
        end
        rel2 = MClassifier_predict(classifier, new_instance)
        if rel1 == rel2
            hits += 1
        else
            misses += 1
        end
    end

    x = hits / n_instancias

    dataset_result = DataFrame(dataset_name =  dataset, acurracy = x)
    append!(dataset_results, dataset_result)
end

println(dataset_results)

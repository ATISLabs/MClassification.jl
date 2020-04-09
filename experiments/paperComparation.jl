###Importing dependencies
using Pkg
Pkg.activate("../")
using DataFrames, MClassification
include("../src/io.jl")


###Setting path of the datasets and getting the their names
path = "../datasets/sinthetic/"
datasets = split(read(`ls $path`, String), "\n")[1:end-1]

dataset_results = DataFrame()
for dataset in datasets
    n_instances = 0
    hits = 0
    misses = 0

    classifier = MClassifier(0.1)

    dataset_file = open(path * dataset , "r")
    MClassifier_initialize(classifier, dataset_file, 150)

    while true
        sample = load_sample(dataset_file)
        if sample == ""
            break
        end
        n_instances += 1

        new_instance = sample[1:end-1]
        real_label = sample[end]

        predicted_label = MClassifier_predict(classifier, new_instance)

        if real_label == predicted_label
            hits += 1
        else
            misses += 1
        end
    end

    x = hits / n_instances

    dataset_result = DataFrame(dataset =  dataset, n_initial_instances = 150 ,n_samples = n_instances, accuracy = round(x, digits=2, RoundDown))
    append!(dataset_results, dataset_result)
end
println(dataset_results)

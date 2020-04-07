using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("mclassifier.jl")

path = "./dataset/sinthetic/"
datasets = ["1CDT.txt", "1CHT.txt", "2CDT.txt", "2CHT.txt"]

# Parameters k, n_features, n_labels, training_set_size, label_position, n_features
for dataset in datasets
    n_instancias = 0
    acertos = 0
    erros = 0
    classifier = MClassifier(0.1)
    classifier.n_features = 2

    dataset_file = open(path * dataset , "r")
    MClassifier_inicialize(classifier, dataset_file, 150)

    while true
        n_instancias += 1
        new_instance = zeros(Float64, classifier.n_features)
        rel1 = load_instance(dataset_file, 2, new_instance)
        if rel1 == ""
            break
        end
        rel2 = MClassifier_predict(classifier, new_instance)
        if rel1 == rel2
            acertos += 1
        else
            erros += 1
        end
    end

    x = acertos / n_instancias

    println(dataset, " - ", x)
    println("Acertos ", acertos)
    println("Erros ", erros)
end

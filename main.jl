include("mclassifier.jl")

path = "./dataset/sinthetic/"
file_name = "1CDT.txt"
dataset_file = open( path * file_name , "r")

# Parameters k, n_features, n_labels, training_set_size, label_position, n_features

classifier = MClassifier()
classifier.r_limit = 0.1
classifier.n_features = 2
classifier.n_mc = 0

MClassifier_inicialize(classifier, dataset_file, 10)
new_instance = zeros(Float64, 1, classifier.n_features)
load_instance(dataset_file, 2, new_instance)
print(new_instance, "\n")
print(MClassifier_predict(classifier, new_instance))

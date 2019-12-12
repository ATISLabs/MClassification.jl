include("mc.jl")
include("dataStructure.jl")
include("io.jl")

mutable struct MClassifier
    n_mc::Int64
    r_limit::Float64
    micro_clusters::Array{MC}
    n_features::Int64

    function MClassifier()
        new()
    end

    function Mclassifier(r_limit::Float64)
        x = new()
        x.n_mc = 0
        x.r_limit = r_limit
        return x
    end
end

function MClassifier_inicialize(classifier::MClassifier, dataset_file::IOStream, n)

    for i=1:n
        instance = zeros(Float64, 1, classifier.n_features)
        label = load_instance(dataset_file, classifier.n_features, instance)
        mc = MC(instance, Int64(label))
        MClassifier_append_MC(classifier, mc)
    end
end

#=
function MClassifier_inicialize(classifier::MClassifier, dataset_file::IOStream, n)
    instance = zeros(Float64, 1, 3)
    k = classifier.n_features + 1
    for i=1:n
        load_instance(dataset_file, classifier.n_features, instance)
        mc = MC(instance[1:classifier.n_features], k)
        MClassifier_append_MC(classifier, mc)
    end
end
=#

function MClassifier_predict(classifier::MClassifier, instance)
    distances = NodeList()
    for x in classifier.micro_clusters
        distances = ordered_push_in_list(distances, x.id, x.label, calc_distance(instance, x.centroid))
    end
    testess = classifier.micro_clusters[distances.id]

    if mc_predict_r(classifier.micro_clusters[distances.id], instance) <= classifier.r_limit
        mc_append_sample(classifier.micro_clusters[distances.id], instance)
    else
        MClassifier_append_MC(classifier, MC(instance, distances.label))
    end
    #println(classifier.micro_clusters[distances.id].centroid)
    #println(distances.distance)
    print_list(distances)
    println(classifier.micro_clusters[distances.id].centroid)
    return distances.label
end
#=
function MClassifier_predict(classifier::MClassifier, sample)
    distances = NodeList()
    instance = zeros(Float64, classifier.n_features)
    load_instance(file, classifier.n_features, instance)
    for x in classifier.micro_clusters
        distances = ordered_push_in_listO1(distances, x.id, x.label, calc_distance(instance, x.centroid))
    end

    if mc_predict_r(classifier.micro_clusters[distances.id], sample) <= classifierr.r_limit
        mc_append_sample(classifier.micro_clusters[distances.id], sample)
    else
        mc.micro_clusters = hcat(mc.micro_clusters, MC(sample))
    end

    return distance.label
end
=#
function MClassifier_append_MC(classifier::MClassifier, micro_cluster::MC)
    classifier.n_mc += 1
    micro_cluster.id = classifier.n_mc
    if classifier.n_mc == 1
        classifier.micro_clusters = [micro_cluster]
    else
        classifier.micro_clusters = vcat(classifier.micro_clusters, micro_cluster)
    end

end

include("mc.jl")
include("io.jl")

using DataStructures

mutable struct MClassifier
    n_mc::Int64
    r_limit::Float64
    micro_clusters::Set{MC}
    n_features::Int64

    function MClassifier()
        new()
    end

    function MClassifier(r_limit::Float64)
        x = new()
        x.n_mc = 0
        x.r_limit = r_limit
        x.micro_clusters = Set{MC}()
        return x
    end
end

function MClassifier_inicialize(classifier::MClassifier, dataset_file::IOStream, n)

    for i=1:n
        instance = zeros(Float64, classifier.n_features)
        label = load_instance(dataset_file, classifier.n_features, instance)
        mc = MC(instance, Int64(label))
        MClassifier_append_MC(classifier, mc)
    end
end

function MClassifier_predict(classifier::MClassifier, instance)
    distances = OrderedDict{Float64, MC}()

    for x in classifier.micro_clusters
        distances[calc_distance(instance, x.centroid)] = x
    end
    distances = sort(distances)


    distances_array = collect(values(distances))
    array_size = length(distances_array)
    micro_cluster = distances_array[1]
    if mc_predict_r(micro_cluster, instance) <= classifier.r_limit
        mc_append_sample(micro_cluster, instance)
    else
        MClassifier_append_MC(classifier, MC(instance, micro_cluster.label))
        count::Int16 = 0
        mcs_farthest = Array{MC, 1}()

        for i in 0:array_size - 1
            if distances_array[array_size - i].label == distances_array[1].label
                count += 1
                append!(mcs_farthest, [distances_array[array_size - i]])
                if count == 2
                    mc_merge(mcs_farthest[1], mcs_farthest[2])
                    delete!(classifier.micro_clusters, mcs_farthest[2])
                    break
                end
            end
        end
    end
    return micro_cluster.label
end

function MClassifier_append_MC(classifier::MClassifier, micro_cluster::MC)
    classifier.n_mc += 1
    push!(classifier.micro_clusters, micro_cluster)
end

function MClassifier_merge_mc(classifier::MClassifier, mc1, mc2)
    mc1 = mc_merge(mc1, mc2)
    delete!(classifier.micro_clusters, mc2)
end

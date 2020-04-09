module MClassification
    include("mc.jl")
    include("io.jl")

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

    function fit(dataset_file::IOStream, n, r_limit::Float64)
        classifier = MClassifier(r_limit)
        for i=1:n
            sample = load_sample(dataset_file)
            label = sample[end]
            mc = MC(sample[1:end-1], Int64(label))
            append!(classifier, mc)
        end
        return classifier
    end
    
    function fit( X::Array{Float64, 2}, Y::Array{Int64, 1}, r_limit::Float64)
        classifier = MClassifier(r_limit)
        for i in 1:length(Y)
            label = Y[i]
            mc = MC(X[i, :], Int64(label))
            append!(classifier, mc)
        end
        return classifier
    end

    function fit( X, Y::Array{Int64, 1}, r_limit::Float64)
        classifier = MClassifier(r_limit)
        for i in 1:length(Y)
            label = Y[i]
            mc = MC(X[i], Int64(label))
            append!(classifier, mc)
        end
        return classifier
    end

    function predict(classifier::MClassifier, instance)
        distances = Array{Any, 1}()
        for micro_cluster in classifier.micro_clusters
            push!(distances, [calc_distance(instance, micro_cluster.centroid), micro_cluster])
        end
        sort!(distances, by = x -> x[1])
        array_size = length(distances)
        micro_cluster = distances[1][2]

        if mc_predict_r(micro_cluster, instance) <= classifier.r_limit
            append!(micro_cluster, instance)
        else
            append!(classifier, MC(instance, micro_cluster.label))
            count = 0
            mcs_farthest = Array{MC, 1}()

            for i in 0:array_size - 1
                if distances[array_size - i][2].label == distances[1][2].label
                    count += 1
                    Base.append!(mcs_farthest, [distances[array_size - i][2]])
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

    function append!(classifier::MClassifier, micro_cluster::MC)
        classifier.n_mc += 1
        push!(classifier.micro_clusters, micro_cluster)
    end

    function MClassifier_merge_mc(classifier::MClassifier, mc1, mc2)
        mc1 = mc_merge(mc1, mc2)
        delete!(classifier.micro_clusters, mc2)
    end
end # module

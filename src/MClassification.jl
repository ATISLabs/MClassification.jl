module MClassification
    using DataFrames
    include("./MCluster.jl")
    include("io.jl")

    mutable struct MClassifier
        n_mc::Int64
        r_limit::Float64
        micro_clusters::Set{MCluster.MC}
        n_features::Int64

        function MClassifier()
            new()
        end

        function MClassifier(r_limit::Float64)
            x = new()
            x.n_mc = 0
            x.r_limit = r_limit
            x.micro_clusters = Set{MCluster.MC}()
            return x
        end
    end

    function fit(dataset_file::IOStream, n, r_limit::Float64)
        classifier = MClassifier(r_limit)
        for i=1:n
            sample = load_sample(dataset_file)
            label = sample[end]
            mc = MCluster.MC(sample[1:end-1], Int64(label))
            append!(classifier, mc)
        end
        return classifier
    end

    function fit( X::Array{Float64, 2}, Y::Array{Int64, 1}, r_limit::Float64)
        classifier = MClassifier(r_limit)
        for i in 1:length(Y)
            label = Y[i]
            mc = MCluster.MC(X[i, :], Int64(label))
            append!(classifier, mc)
        end
        return classifier
    end

    function fit( X::DataFrame, Y::Array{Int64, 1}, r_limit::Float64)
        classifier = MClassifier(r_limit)

        for i in 1:length(Y)
            label = Y[i]
            mc = MCluster.MC(X[i, :], label)
            append!(classifier, mc)
        end
        return classifier
    end

    function fit( X::DataFrame, Y::CategoricalArray, r_limit::Float64)
        classifier = MClassifier(r_limit)

        for i in 1:length(Y)
            label = Y[i]
            mc = MCluster.MC(X[i, :], label)
            append!(classifier, mc)
        end
        return classifier
    end

    function predict(classifier::MClassifier, instance::Array{T, 1}) where {T<:Number}
        distances = Array{Any, 1}()
        for micro_cluster in classifier.micro_clusters
            Base.push!(distances, [MCluster.calc_distance(instance, micro_cluster.centroid), micro_cluster])
        end
        sort!(distances, by = x -> x[1])
        array_size = length(distances)
        micro_cluster = distances[1][2]

        if MCluster.predict_r(micro_cluster, instance) <= classifier.r_limit
            MCluster.append!(micro_cluster, instance)
        else
            append!(classifier, MCluster.MC(instance, micro_cluster.label))
            count = 0
            mcs_farthest = Array{MCluster.MC, 1}()

            for i in 0:array_size - 1
                if distances[array_size - i][2].label == distances[1][2].label
                    count += 1
                    Base.append!(mcs_farthest, [distances[array_size - i][2]])
                    if count == 2
                        merge_farthest(classifier, mcs_farthest[1], mcs_farthest[2])
                        break
                    end
                end
            end
        end
        return micro_cluster.label
    end

    function predict(classifier::MClassifier, instances::Array{T, 2}) where {T<:Number}
        y_predicted = Array{Any, 1}()
        print(size(instances)[1])
        test = false

        for k in 1:size(instances)[1]
            instance = instances[k, :]
            distances = Array{Any, 1}()
            for micro_cluster in classifier.micro_clusters
                push!(distances, [MCluster.calc_distance(instance, micro_cluster.centroid), micro_cluster])
            end
            sort!(distances, by = x -> x[1])
            array_size = length(distances)
            micro_cluster = distances[1][2]

            if MCluster.predict_r(micro_cluster, instance) <= classifier.r_limit
                MCluster.append!(micro_cluster, instance)
            else
                append!(classifier, MCluster.MC(instance, micro_cluster.label))
                count = 0
                mcs_farthest = Array{MCluster.MC, 1}()

                for i in 0:array_size - 1
                    if distances[array_size - i][2].label == distances[1][2].label
                        count += 1
                        Base.append!(mcs_farthest, [distances[array_size - i][2]])
                        if count == 2
                            merge_farthest(classifier, mcs_farthest[1], mcs_farthest[2])
                            break
                        end
                    end
                end
            end
            Base.append!(y_predicted, micro_cluster.label)
        end
        return y_predicted
    end

    function append!(classifier::MClassifier, micro_cluster::MCluster.MC)
        classifier.n_mc += 1
        push!(classifier.micro_clusters, micro_cluster)
    end

    function merge_farthest(classifier::MClassifier, mc_a::MCluster.MC, mc_b::MCluster.MC)
        mc_a = MCluster.merge(mc_a, mc_b)
        delete!(classifier.micro_clusters, mc_b)
    end
end # module

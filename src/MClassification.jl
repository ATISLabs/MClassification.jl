module MClassification
    using DataFrames, MLJBase, Distances
    export euclidean_distance

    euclidean_distance = Distances.Euclidean()
    include("./MCluster.jl")

    mutable struct MClassifier <: MLJBase.Deterministic
        r_limit::Float64
        n_features::Int64
        metric

        function MClassifier(;r_limit = 0.1, metric = euclidean_distance)
            new_MClassifier = new()
            new_MClassifier.r_limit = r_limit
            new_MClassifier.metric = metric
            return new_MClassifier
        end
    end

    function fit(model::MClassifier, verbosity::Int, X::Array{T, 2}, Y::CategoricalArray) where {T<:Number}
        fitresult = Set{MCluster.MC}()
        for i in 1:length(Y)
            mc = MCluster.MC(X[i, :], Y[i])
            push!(fitresult, mc)
        end
        cache = nothing
        report = nothing
        return fitresult, cache, report
    end


    function fit(model::MClassifier, verbosity::Int, X, Y::CategoricalArray)
        x = Matrix(X)
        fitresult = Set{MCluster.MC}()
        for i in 1:length(Y)
            label = Y[i]
            mc = MCluster.MC(x[i, :], label)
            push!(fitresult, mc)
        end
        cache = nothing
        report = nothing
        return fitresult, cache, report
    end

    function predict(classifier::MClassifier, fitresult, instance::Array{T, 1}) where {T<:Number}
        distances = Array{Any, 1}()
        for micro_cluster in fitresult
            Base.push!(distances, [classifier.metric(instance, micro_cluster.centroid), micro_cluster])
        end

        sort!(distances, by = x -> x[1])
        array_size = length(distances)
        micro_cluster = distances[1][2]

        if MCluster.predict_r(micro_cluster, instance) <= classifier.r_limit
            MCluster.append!(micro_cluster, instance)
        else
            push!(fitresult, MCluster.MC(instance, micro_cluster.label))
            count = 0
            mcs_farthest = Array{MCluster.MC, 1}()

            for i in 0:array_size - 1
                if distances[array_size - i][2].label == distances[1][2].label
                    count += 1
                    push!(mcs_farthest, distances[array_size - i][2])
                    if count == 2
                        merge_farthest(fitresult, mcs_farthest[1], mcs_farthest[2])
                        break
                    end
                end
            end
        end
        return micro_cluster.label
    end

    function predict(classifier::MClassifier, fitresult, samples::Array{T, 2}) where {T<:Number}
        return [predict(classifier, fitresult, samples[i, :]) for i in 1:nrows(samples)]
    end

    function predict(classifier::MClassifier, fitresult, samples) where {T<:Number}
        X = Matrix(samples)
        predict(classifier, fitresult, X)
    end

    function append!(classifier::MClassifier, micro_cluster::MCluster.MC)
        push!(classifier.micro_clusters, micro_cluster)
    end

    function merge_farthest(micro_clusters::Set, mc_a::MCluster.MC, mc_b::MCluster.MC)
        mc_a = MCluster.merge(mc_a, mc_b)
        delete!(micro_clusters, mc_b)
    end

    include("./MLJBase.jl")
    include("./EasyStream.jl")
end # module

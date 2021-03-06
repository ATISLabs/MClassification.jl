module MCluster
using DataFrames

    mutable struct MC
        label
        n::Int64
        ls::Array{Float64}
        ss::Array{Float64}
        centroid::Array{Float64}

        function MC()
            x = new()
        end

        function MC(sample::Array{T, 1}, label) where {T<:Number}
            v_size = length(sample)
            x = new()
            x.n = 1
            x.label = label
            x.ls = zeros(T, v_size)
            x.ss = zeros(T, v_size)
            for i = 1:v_size
                x.ls[i] = x.ls[i] + sample[i]
                x.ss[i] = x.ss[i] + sample[i] ^ 2
            end
            x.centroid = copy(sample)
            return x
        end
    end

    function mc_set_centroid(mc::MC)
        mc.centroid = mc.ls / mc.n
    end

    function predict_r(mc::MC, sample::Array{T}) where {T<:Number}
        ss = mc.ss + sample .^2
        ls = mc.ls + sample
        n = mc.n + 1
        return abs(sum((ss / n) - (ls / n).^ 2)) ^ (1/2)
    end

    function append!(mc::MC, sample::Array{T}) where {T<:Number}
        mc.n = mc.n + 1
        mc.ls = mc.ls + sample
        mc.ss = mc.ss + sample .^ 2
        mc_set_centroid(mc)
    end

    function merge(mc_a::MC, mc_b::MC)
        new_mc = MC()
        new_mc.n = mc_a.n + mc_b.n
        new_mc.ls = mc_a.ls + mc_b.ls
        new_mc.ss = mc_a.ss + mc_b.ss
        mc_set_centroid(new_mc)
        return new_mc

    end
end #module

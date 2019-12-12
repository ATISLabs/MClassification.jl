mutable struct MC
    id::Int64
    n::Int64
    label::Int64
    ls::Array{Float64}
    ss::Array{Float64}
    samples::Array{Float64}
    r::Float64
    centroid::Array{Float64}

    function MC(sample::Array{Float64}, label::Int64)
        v_size = size(sample)[2]
        x = new()
        x.n = 1
        x.label = label
        x.ls = zeros(Float64, 1, v_size)
        x.ss = zeros(Float64, 1, v_size)
        for i = 1:v_size
            x.ls[i] = x.ls[i] + sample[i]
            x.ss[i] = x.ss[i] + sample[i] ^ 2
        end
        x.centroid = sample
        x.samples = sample
        return x
    end
end

function mc_set_r(mc::MC)
    mc.r = abs(sum((mc.ss / mc.n) - (mc.ls / mc.n).^ 2)) ^ (1/2)
end

function mc_set_centroid(mc::MC)
    mc.centroid = mc.ls / mc.n
end

function mc_predict_r(mc::MC, sample::Array{Float64})

    ss = mc.ss + sample .^2
    ls = mc.ls + sample
    n = mc.n + 1
    return abs(sum((ss / n) - (ls / n).^ 2)) ^ (1/2)
end

function mc_append_sample(mc::MC, sample::Array{Float64})
    v_size = size(sample)[2]
    mc.n = mc.n + 1

    mc.ls = mc.ls + sample
    mc.ss = mc.ss + sample .^ 2

    mc.samples = [mc.samples; sample]
    mc_set_r(mc)
    mc_set_centroid(mc)
end

function calc_distance(a::Array{Float64}, b::Array{Float64})

    return sum((a - b).^2) ^ (1/2)
end

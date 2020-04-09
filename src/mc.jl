mutable struct MC
    label::Int64
    n::Int64
    ls::Array{Float64}
    ss::Array{Float64}
    r::Float64
    centroid::Array{Float64}

    function MC(sample::Array{Float64}, label::Int64)
        v_size = length(sample)
        x = new()
        x.n = 1
        x.label = label
        x.ls = zeros(Float64, v_size)
        x.ss = zeros(Float64, v_size)
        for i = 1:v_size
            x.ls[i] = x.ls[i] + sample[i]
            x.ss[i] = x.ss[i] + sample[i] ^ 2
        end
        x.centroid = copy(sample)
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

function append!(mc::MC, sample::Array{Float64})
    mc.n = mc.n + 1
    mc.ls = mc.ls + sample
    mc.ss = mc.ss + sample .^ 2

    mc_set_r(mc)
    mc_set_centroid(mc)
end

function calc_distance(a::Array{Float64}, b::Array{Float64})
    return sum((a - b).^2) ^ (1/2)
end

function mc_merge(mc_1::MC, mc_2::MC)
    mc_1.n = mc_1.n + mc_2.n
    mc_1.ls = mc_1.ls + mc_2.ls
    mc_1.ss = mc_1.ss + mc_2.ss
    mc_set_centroid(mc_1)

end

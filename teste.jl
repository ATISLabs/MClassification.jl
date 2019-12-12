include("mc.jl")

v = MC([1.0 3.0 3.0], 3)
v2 = MC([2.0 1.0 4.5], 2)

dump(v)
dump(v2)

v = hcat(v, v2)
dump(v)

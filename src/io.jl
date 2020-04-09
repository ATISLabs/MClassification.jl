function load_instances(dataset_file::IOStream, n_features::Int64, training_set_size::Int64, training_set::Array{Float64})
    for i=1:training_set_size
        j = 1
        while j <= n_features
            training_set[i, j] = parse(Float64, readuntil(dataset_file, ','))
            j += 1
        end
        training_set[i, j] = parse(Float64, readuntil(dataset_file, '\n'))
    end
end

function load_sample(dataset_file::IOStream)
    line = readuntil(dataset_file, "\n")
    if line != ""
        return [parse(Float64,x) for x in split(line, ",")]
    end
        return ""
end

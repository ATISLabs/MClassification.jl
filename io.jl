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

function load_instance(dataset_file::IOStream, n_features::Int64, instance::Array{Float64})
    j = 1
    while j <= n_features
        instance[j] = parse(Float64, readuntil(dataset_file, ','))
        j += 1
    end
    return parse(Float64, readuntil(dataset_file, '\n'))
end

function prediction(training_set, instance, k)

end

function print_test()
    println("Deu certo")
end

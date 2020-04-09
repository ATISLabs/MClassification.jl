function load_sample(dataset_file::IOStream)
    line = readuntil(dataset_file, "\n")
    if line != ""
        return [parse(Float64,x) for x in split(line, ",")]
    end
        return ""
end

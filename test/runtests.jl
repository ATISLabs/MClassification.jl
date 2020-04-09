using MClassification, Test
include("../src/io.jl")
@testset "Fit - Array" begin
    initial_samples = 4
    X = [5.2603 0.40807;
    0.51512 -0.7065;
    4.721 4.5322;
    0.36285 1.0328;
    0.89978 1.6674;
    2.7659 0.74216;
    1.4156 2.2328;
    1.0293 0.12319;
    3.3528 3.261]

    Y = [2, 1, 2, 1, 1, 1, 1, 1, 2]
    classifier = MClassification.fit(X[1:initial_samples, :], Y[1:initial_samples], 0.1)

    for i in initial_samples+1:length(Y)
        @test MClassification.predict(classifier, X[i, :]) == Y[i]
    end
    @test initial_samples == length(classifier.micro_clusters)
end

@testset "Fit - IOSTREAM" begin
    initial_samples = 4
    dataset_file = open("test.txt")
    classifier = MClassification.fit(dataset_file, 4, 0.1)

    while true
        sample = load_sample(dataset_file)
        if sample == ""
            break
        end
        @test MClassification.predict(classifier, sample[1:end-1]) == sample[end]
    end
    @test initial_samples == length(classifier.micro_clusters)
end

@testset "Load Sample - IO" begin
    dataset_file = open("test.txt")
    X = [5.2603 0.40807 2;
    0.51512 -0.7065 1;
    4.721 4.5322 2;
    0.36285 1.0328 1;
    0.89978 1.6674 1;
    2.7659 0.74216 1;
    1.4156 2.2328 1;
    1.0293 0.12319 1;
    3.3528 3.261 2]

    i = 0

    while true
        i += 1
        sample = load_sample(dataset_file)
        if sample == ""
            break
        end
        @test sample == X[i, :]
    end
end

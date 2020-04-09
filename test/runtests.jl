using MClassification, Test
@testset "Classification Test" begin
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

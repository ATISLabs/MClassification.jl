using MClassification, Test, DataFrames

@testset "Predict Array{Number, 2}" begin
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
    y = MClassification.categorical([2, 1, 2, 1, 1, 1, 1, 1, 2])
    model = MClassification.MClassifier(r_limit=0.1)

    fitresult, cache, report = MClassification.fit(model, 0, X[1:initial_samples, :], y[1:initial_samples])

    y_predicted = MClassification.predict(model, fitresult, X[initial_samples+1:end, :])

    for i in 1:length(y)-initial_samples
        @test y_predicted[i] == y[initial_samples+i]
    end

    @test initial_samples == length(fitresult)
end

@testset "Prediction - DataFrame" begin
    initial_samples = 4
    X = [5.2603 0.40807;
    0.51512 -0.7065;
    4.721 4.5322;
    0.36285 1.0328;
    0.89978 1.6674;
    2.7659 0.74216;
    1.4156 2.2328;
    1.0293 0.12319;
    3.3528 3.261] |> DataFrame

    y = MClassification.categorical([2, 1, 2, 1, 1, 1, 1, 1, 2])
    model = MClassification.MClassifier(r_limit=0.1)

    fitresult, cache, report = MClassification.fit(model, 0, X[1:initial_samples, :], y[1:initial_samples])

    y_predicted = MClassification.predict(model, fitresult, X[initial_samples+1:end, :])

    for i in 1:length(y)-initial_samples
        @test y_predicted[i] == y[initial_samples+i]
    end

    @test initial_samples == length(fitresult)
end


@testset "Append MC" begin
    samples = [[3, 3, 3], [2, 2, 2]]
    mc = MClassification.MCluster.MC([1, 1, 1], 1)
    centroid = [1, 1, 1]
    ls = [1, 1, 1]
    ss = [1, 1, 1]
    n = 1
    @test mc.centroid == centroid
    @test mc.ls == ls
    @test mc.ss == ss
    @test mc.n == n

    for sample in samples
        r_predicted = MClassification.MCluster.predict_r(mc, sample)

        MClassification.MCluster.append!(mc, sample)
        n += 1
        ls = ls .+ sample
        ss = ss + sample.*sample
        centroid = ls ./ n

        @test r_predicted == abs(sum((ss / n) - (ls / n).^ 2)) ^ (1/2)
        @test mc.centroid == centroid
        @test mc.ls == ls
        @test mc.ss == ss
        @test mc.n == n
    end
end

@testset "Evaluate - MLJ" begin
    # Evaluating model using evaluation of MLJ Library

    X        = [5.2603 0.40807; 0.51512 -0.7065; 4.721 4.5322;
                0.36285 1.0328; 0.89978 1.6674; 2.7659 0.74216;
                1.4156 2.2328; 1.0293 0.12319; 3.3528 3.261]
    y        = MClassification.categorical([2, 1, 2, 1, 1, 1, 1, 1, 2])

    train    = 1:4
    test     = 5:MClassification.nrows(X)

    model    = MClassification.MClassifier(r_limit=0.1)
    model    = MClassification.machine(model, X, y)

    @test MClassification.evaluate!(model, resampling=[(train, test)], measure=MClassification.accuracy)[:measurement][1] == 1.

end

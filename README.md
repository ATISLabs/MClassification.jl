# MClassification.jl

[![Build Status](https://travis-ci.com/Conradox/MClassification.jl.svg?branch=master)](https://travis-ci.com/Conradox/MClassification.jl)
[![Codecov](https://codecov.io/gh/Conradox/MClassification.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Conradox/MClassification.jl)
[![Coverage Status](https://coveralls.io/repos/github/Conradox/MClassification.jl/badge.svg?branch=master)](https://coveralls.io/github/Conradox/MClassification.jl?branch=master)

A developed algorithm of classification using Micro-Cluster representation from online clustering algorithms to be performed in a incremental concept drift environment. This algorithm has a competitive performance when put against state-of-art algorithms such as SCARGC. Moreover, it only has one hyper-parameter turning into your calibration not difficult.

Install
=======

    Pkg.add("MClassification")

Using
=====

    using MClassification

    Examples
    ========

        using MClassification

        # Learning and classifying using a small sample set

        X_train             = [5.2603 0.40807; 0.51512 -0.7065; 4.721 4.5322;
                               0.36285 1.0328; 0.89978 1.6674; 2.7659 0.74216]

        Y_train             = MClassification.categorical([2, 1, 2, 1, 1, 1])
        X_test              = [1.4156 2.2328; 1.0293 0.12319; 3.3528 3.261]
        Y_test              = MClassification.categorical([1, 1, 2])


        model               = MClassification.MClassifier(r_limit=0.1)
        fitresult, _ , _    = MClassification.fit(model, 0, X_train, Y_train)

        y_hat = MClassification.predict(model, fitresult, X_test)
        print("[MClassification] Accuracy : $(MClassification.accuracy(y_hat, Y_test))")


        # Evaluating model using evaluation of MLJ Library

        X        = [5.2603 0.40807; 0.51512 -0.7065; 4.721 4.5322;
                    0.36285 1.0328; 0.89978 1.6674; 2.7659 0.74216;
                    1.4156 2.2328; 1.0293 0.12319; 3.3528 3.261]
        y        = MClassification.categorical([2, 1, 2, 1, 1, 1, 1, 1, 2])

        train    = 1:4
        test     = 5:MClassification.nrows(X)

        model    = MClassification.MClassifier(r_limit=0.1)
        model    = MClassification.machine(model, X, y)

        println(MClassification.evaluate!(model, resampling=[(train, test)], measure=accuracy))





References
=======
* MClassification
   * V. M. A. Souza, D. F. Silva, G. E. A. P. A. Batista and J. Gama, "Classification of Evolving Data Streams with Infinitely Delayed Labels," 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA), Miami, FL, 2015, pp. 214-219.

* Micro Cluster
    * Aggarwal, Charu & Han, Jiawei & Wang, Jianyong & Yu, Philip & Watson, T. & Ctr, Resch. (2003). A Framework for Clustering Evolving Data Streams.

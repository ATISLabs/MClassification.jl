# MClassification.jl

[![Build Status](https://travis-ci.com/Conradox/MClassification.jl.svg?branch=master)](https://travis-ci.com/Conradox/MClassification.jl)
[![Codecov](https://codecov.io/gh/Conradox/MClassification.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Conradox/MClassification.jl)
[![Coveralls](https://coveralls.io/repos/github/Conradox/MClassification.jl/badge.svg?branch=master)](https://coveralls.io/github/Conradox/MClassification.jl?branch=master)

MClassification.jl
======

A developmented algorithm of classification using Micro-Cluster representation from online clustering algorithms to be performed in a incremental concept drift environment. This algorithm has a competitive performance regard on state-of-art algorithms as SCARGC. Besides, it only has a hyper-parameter turned your calibrate very easy.

Install
=======

    Pkg.add("MClassification")

Using
=====

    using MClassification

    Examples
    ========

        using MClassification

        # learning a single target
        X_train        = [1 2; 2 4; 4 6.0]
        Y_train        = [4; 6; 8.0]
        X_test         = [6 8; 8 10; 10 12.0]
        Y_test         = [10; 12; 14.0]

        classifier     = MClassification.fit(X_train, Y_train, 0.1)
        Y_pred         = MClassification.predict(model, X_test)

        print("[PLS1] mae error : $(mean(abs.(Y_test .- Y_pred)))")

References
=======
* MClassification
   * V. M. A. Souza, D. F. Silva, G. E. A. P. A. Batista and J. Gama, "Classification of Evolving Data Streams with Infinitely Delayed Labels," 2015 IEEE 14th International Conference on Machine Learning and Applications (ICMLA), Miami, FL, 2015, pp. 214-219.

* Micro Cluster
    * Aggarwal, Charu & Han, Jiawei & Wang, Jianyong & Yu, Philip & Watson, T. & Ctr, Resch. (2003). A Framework for Clustering Evolving Data Streams.

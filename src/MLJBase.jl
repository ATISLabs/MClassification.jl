function MLJBase.fit(model::MClassifier, verbosity::Int, X, Y)
    return fit(model, verbosity, X, Y)
end

function MLJBase.predict(classifier::MClassifier, fitresult, instance)
    return update_predict(classifier, fitresult, instance)
end

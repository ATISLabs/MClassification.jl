function EasyStream.fit(model::MClassifier, verbosity::Int, X, Y)
    return fit(model, verbosity, X, Y)
end

function EasyStream.predict(classifier::MClassifier, fitresult, instance)
    return predict(classifier, fitresult, instance)
end

function EasyStream.updatePredict(classifier::MClassifier, fitresult, instance)
    return update_predict(classifier, fitresult, instance)
end

"""
Used in gp.jl.
Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

include("problems.jl")

struct Sequential_dT <: SequentialProblem
    variable::String #"T" or "wT"
    Δt::Number # assumes constant time interval between all timesteps
    # scaling::Scaling
end

struct Sequential_T <: SequentialProblem
    variable::String # "T" or "wT"
    # scaling::Scaling
end

struct Sequential_wT <: SequentialProblem
    variable::String # "T" or "wT"
    # scaling::Scaling
end

# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
# | Sequential_dT                              |
# |                                            |
# |   predictor       target                   |
# |   T[i] --model--> (T[i+1]-T[i])/Δt ≈ ∂t(T) |
# |                                            |
# |   model output                             |
# |   model(T[i])                              |
# |                                            |
# |   prediction                               |
# |   predicted T[i+1] = model(T[i])Δt + T[i]  |
# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

"""
get_predictors_targets(vavg::Array, problem::Sequential_T)
----- Description
    Returns x and y, the predictor and target pairs from which to extract the training and verification data sets for "T" profiles.

    predictors --model--> targets ("truth")
          T[i] --model--> (T[i+1]-T[i])/Δt ≈ ∂t(T)

----- Arguments
- 'vavg': (Array)             Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
- 'problem': (Residual_T)     Sequential_T object associated with the data (output of get_problem)
"""
function get_predictors_targets(vavg::Array, problem::Residual_T)
    # vavg should be scaled!
    predictors = vavg[1:end-1] # T[i] for i = 1, ..., Nt-1
    predictions = (vavg[2:end] - predictors) / problem.Δt # (T[i+1]-T[i])/Δt
    return (predictors, targets)
end

"""
----- Description
Takes in a scaled predictor, T[i], the scaled model output on the predictor, model(T[i]), and a Sequential_T object.
Returns the scaled prediction (predicted temperature profile), T[i+1], computed from T[i] and model(T[i]) by

    predicted T[i+1] = model(T[i])Δt + T[i]

----- Arguments
'scaled_predictor': (Array)            T[i], the scaled predictor for a temperature profile (T[i])
'scaled_model_output': (Array)         model(T[i]), the scaled prediction for a temperature profile (G(T[i]) or NN(T[i]))
'problem': (Problem)
"""
function postprocess_prediction(scaled_predictor, scaled_model_output, problem::Residual_T)
    return scaled_model_output * problem.Δt + scaled_predictor
end

# *--*--*--*--*--*--*--*--*--*--*
# | Sequential_T                |
# |   T[i] --model--> T[i+1]    |
# *--*--*--*--*--*--*--*--*--*--*

"""
get_predictors_targets(vavg::Array, problem::Sequential_T)
----- Description
    Returns x and y, the predictors and target predictions from which to extract the training and verification data for "T" profiles.

    predictors --model--> targets ("truth")
          T[i] --model--> T[i+1]

----- Arguments
- 'vavg': (Array)               Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
- 'problem': (Sequential_T)     Sequential_T object associated with the data (output of get_problem)
"""
function get_predictors_targets(vavg::Array, problem::Sequential_T)
    # vavg should be scaled!
    predictors = vavg[1:end-1] # T[i] for i = 1, ..., Nt-1
    targets = vavg[2:end] # T[i+1]
    return (predictors, targets)
end

"""
----- Description
Takes in a scaled predictor, T[i], the scaled GP prediction on the predictor, G(T[i]), and a Sequential_T object.
Returns the predicted temperature profile, T[i+1], computed from T[i] and G(T[i]) by

           prediction = model(predictor)
     predicted T[i+1] = model(T[i])

----- Arguments
'scaled_predictor': (Array)            T[i], the scaled predictor for a temperature profile
'scaled_prediction': (Array)           model(T[i), the scaled prediction for a temperature profile
'problem': (Sequential_T)
"""
function postprocess_prediction(scaled_predictor, scaled_prediction, problem::Sequential_T)
    return scaled_prediction
end

# *--*--*--*--*--*--*--*--*--*--*
# | Sequential_wT               |
# |   wT[i] --model--> wT[i+1]  |
# *--*--*--*--*--*--*--*--*--*--*

"""
get_predictors_targets(vavg::Array, problem::Sequential_wT)
----- Description
    Returns x and y, the predictors and targets from which to extract the training and verification data for "wT" profiles.

    predictors --model--> targets ("truth")
         wT[i] --model--> wT[i+1]

----- Arguments
- 'vavg': (Array)                  Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
- 'problem': (Sequential_wT)     Sequential_wT object associated with the data (output of get_problem)
"""
function get_predictors_targets(vavg::Array, problem::Sequential_wT)
    # vavg should be scaled
    predictors = vavg[1:end-1] # wT[i] for i = 1, ..., Nt-1
    targets = vavg[2:end] # wT[i+1]
    return (predictors, targets)
end

"""
----- Description
Takes in a scaled predictor, wT[i], the scaled model prediction on the predictor, model(wT[i]), and a Sequential_wT object.
Returns the temperature profile, T[i+1], computed from model(T[i]) by

           prediction = model(predictor)
    predicted wT[i+1] = model(wT[i])

----- Arguments
'scaled_predictor': (Array)                        wT[i], the scaled predictor for a wT profile
'scaled_prediction': (Array)    predicted wT[i+1], the scaled model prediction for a wT profile
'problem': (Sequential_wT)          Sequential_wT object associated with the data
"""
function postprocess_prediction(scaled_predictor, scaled_prediction, problem::Sequential_wT)
    return scaled_prediction # do nothing
end

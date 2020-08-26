"""
Used in gp.jl.
Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

include("problems.jl")

# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
# | Residual_KPP                                     |
# |                                                  |
# |   predictor            target                    |
# |   KPP(T[i]) --model--> T[i] - KPP(T[i])          |
# |                                                  |
# |   model output                                   |
# |   model(KPP(T[i]))                               |
# |                                                  |
# |   predicted T[i] = model(KPP(T[i])) + KPP(T[i])  |
# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

"""
get_predictors_targets(vavg::Array, problem::Residual_KPP)
----- Description
    Returns x and y, the predictors and target predictions from which to extract the training and verification data for "T" profiles.

    predictors --model--> targets ("truth")
          T[i] --model--> T[i+1]

----- Arguments
- `vavg`: (Array)               Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
-  `problem`: (Residual_KPP)    Residual_KPP object associated with the data (output of get_problem)
"""
function get_predictors_targets(vavg::Array, problem::Residual_KPP)
    # vavg should be scaled!
    predictors = vavg[1:end-1] # T[i] for i = 1, ..., Nt-1
    targets = vavg[2:end] # T[i+1]
    return (predictors, targets)
end

"""
----- Description
Takes in a scaled predictor, T[i], the scaled GP prediction on the predictor, G(T[i]), and a Residual_KPP object.
Returns the predicted temperature profile, T[i+1], computed from T[i] and G(T[i]) by

           prediction = model(predictor)
     predicted T[i+1] = model(T[i])

----- Arguments
`scaled_predictor`: (Array)            T[i], the scaled predictor for a temperature profile
`scaled_prediction`: (Array)           model(T[i), the scaled prediction for a temperature profile
`problem`: (Residual_KPP)
"""
function postprocess_prediction(scaled_predictor, scaled_prediction, problem::Sequential_T)
    return scaled_prediction
end

# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
# | Residual_TKE                                     |
# |                                                  |
# |   predictor            target                    |
# |   TKE(T[i]) --model--> T[i] - TKE(T[i])          |
# |                                                  |
# |   model output                                   |
# |   model(TKE(T[i]))                               |
# |                                                  |
# |   predicted T[i] = model(TKE(T[i])) + TKE(T[i])  |
# *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

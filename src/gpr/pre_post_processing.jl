"""
Used in GP1.jl.
Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

include("scalings.jl")

abstract type DataProcessor end

struct ResidualTprocessor <: DataProcessor
    # scaling::Scaling
end

struct Tprocessor <: DataProcessor
    Δt::Number # assumes constant time interval between all timesteps
    # scaling::Scaling
end

struct wTprocessor <: DataProcessor
    # scaling::Scaling
end

"""
get_processor(data::ProfileData, timeseries, V_name)
----- Description
    Creates an instance of a DataProcessor struct depending on whether the variable is T or wT.
----- Arguments
- 'data': (OceananigansData)       struct containing data from simulation.
- 'timeseries': (Array)            simulation timeseries [s]
- 'V_name': (String)               "T" for temperature profile or "wT" for temperature flux
"""
function get_processor(data::OceananigansData, timeseries, V_name)
    # if merging data from multiple files, make sure to call this function separately
    # for each different file
    α = 2e-4
    g = 9.80665
    # b_initial = 𝒟.T[:,1] .* α*g
    # approximate initial buoyancy gradient N², where b = N²z + 20*α*g
    #                                                 T = N²z/(αg) + 20
    T_initial = data.T[:,1]
    N² = (T_initial[1] - 20)*α*g / data.z[1]

    if V_name == "T"
        Δt = (timeseries[2]-timeseries[1]) / N²
        return Tprocessor(Δt)
    elseif V_name == "wT"
        return wTprocessor()
    else
        throw(error())
    end
end

# *--*--*--*--*--*
# | residual T   |
# *--*--*--*--*--*
# T[i] --G--> OceanTurb(T[i]) - T[i]

"""
get_predictors_predictions(vavg::Array, processor::ResidualTprocessor)
----- Description
    Returns x and y, the predictors and predictions from which to extract the training and verification data for "T" profiles.

    predictors --model--> predictions
          T[i] --model--> (T[i+1]-T[i])/Δt ≈ ∂t(T)

----- Arguments
- 'vavg': (Array)               Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
- 'processor': (Tprocessor)     Tprocessor object associated with the data (output of get_processor)
"""
function get_predictors_predictions(vavg::Array, processor::Tprocessor)
    # vavg should be scaled!
    predictors = vavg[1:end-1] # T[i] for i = 1, ..., Nt-1
    predictions = (vavg[2:end] - predictors) / processor.Δt # G(T[i]) = (T[i+1]-T[i])/Δt
    return (predictors, predictions)
end

"""
----- Description
Takes in a scaled predictor, T[i], the scaled GP prediction on the predictor, G(T[i]), and a Tprocessor object.
Returns the temperature profile, T[i+1], computed from T[i] and G(T[i]) by

T[i+1] = G(T[i])Δt + T[i]

----- Arguments
'Ti_scaled': (Array)            the scaled predictor for a temperature profile
'GTi_scaled': (Array)           the scaled prediction for a temperature profile
'processor': (DataProcessor)
"""
function postprocess_prediction(Ti_scaled, GTi_scaled, processor::ResidualTprocessor)
    return GTi_scaled*processor.Δt + Ti_scaled
end

# *--*--*
# | T   |
# *--*--*
# T[i] --G--> (T[i+1]-T[i])/Δt ≈ ∂t(T)

"""
get_predictors_predictions(vavg::Array, processor::Tprocessor)
----- Description
    Returns x and y, the predictors and predictions from which to extract the training and verification data for "T" profiles.

    predictors --model--> predictions
          T[i] --model--> (T[i+1]-T[i])/Δt ≈ ∂t(T)

----- Arguments
- 'vavg': (Array)               Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
- 'processor': (Tprocessor)     Tprocessor object associated with the data (output of get_processor)
"""
function get_predictors_predictions(vavg::Array, processor::Tprocessor)
    # vavg should be scaled!
    predictors = vavg[1:end-1] # T[i] for i = 1, ..., Nt-1
    predictions = (vavg[2:end] - predictors) / processor.Δt # G(T[i]) = (T[i+1]-T[i])/Δt
    return (predictors, predictions)
end

"""
----- Description
Takes in a scaled predictor, T[i], the scaled GP prediction on the predictor, G(T[i]), and a Tprocessor object.
Returns the temperature profile, T[i+1], computed from T[i] and G(T[i]) by

T[i+1] = G(T[i])Δt + T[i]

----- Arguments
'Ti_scaled': (Array)            the scaled predictor for a temperature profile
'GTi_scaled': (Array)           the scaled prediction for a temperature profile
'processor': (DataProcessor)
"""
function postprocess_prediction(Ti_scaled, GTi_scaled, processor::Tprocessor)
    return GTi_scaled*processor.Δt + Ti_scaled
end

# *--*--*
# | wT  |
# *--*--*

"""
get_predictors_predictions(vavg::Array, processor::wTprocessor)
----- Description
    Returns x and y, the predictors and predictions from which to extract the training and verification data for "wT" profiles.

    predictors --model--> predictions
         wT[i] --model--> wT[i+1]

----- Arguments
- 'vavg': (Array)               Nt-length array of D-length vectors. Data from which to extract x and y, the predictors and corresponding predictions.
- 'processor': (Tprocessor)     Tprocessor object associated with the data (output of get_processor)
"""
function get_predictors_predictions(vavg::Array, processor::wTprocessor)
    # vavg should be scaled
    predictors = vavg[1:end-1] # wT[i] for i = 1, ..., Nt-1
    predictions = vavg[2:end] # wT[i+1]
    return (predictors, predictions)
end

"""
----- Description
Takes in a scaled predictor, wT[i], the scaled GP prediction on the predictor, G(wT[i]), and a wTprocessor object.
Returns the temperature profile, T[i+1], computed from G(T[i]) by

wT[i+1] = G(wT[i])

----- Arguments
'wTi': (Array)                     the scaled predictor for a wT profile
'scaled_prediction': (Array)       the scaled prediction for a wT profile
'processor': (DataProcessor)       wTprocessor object associated with the data
"""
function postprocess_prediction(wTi, scaled_prediction, processor::wTprocessor)
    return scaled_prediction # do nothing
end

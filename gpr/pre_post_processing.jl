"""
Used in GP1.jl.
Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

include("scalings.jl")

abstract type DataProcessor end

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

# *--*--*
# | T   |
# *--*--*
# T[i] --G--> (T[i+1]-T[i])/Δt ≈ ∂t(T)

function get_predictors_predictions(vavg::Array, processor::Tprocessor)
    # vavg should be scaled!
    # if merging data from multiple files, make sure to call this function separately
    # for each different file
    predictors = vavg[1:end-1] # T[i] for i = 1, ..., Nt-1
    predictions = (vavg[2:end] - predictors) / processor.Δt # G(T[i]) = (T[i+1]-T[i])/Δt
    return (predictors, predictions)
end

function postprocess_prediction(Ti_scaled, GTi_scaled, data_processor::Tprocessor)
    # Takes in a scaled predictor, T[i], a scaled prediction, G(T[i]), and a Tprocessor object.
    # T[i+1] = G(T[i])Δt + T[i]
    return GTi_scaled*data_processor.Δt + Ti_scaled
end

# *--*--*
# | wT  |
# *--*--*
# wT[i] --G--> wT[i+1]

function get_predictors_predictions(vavg::Array, processor::wTprocessor)
    # vavg should be scaled
    predictors = vavg[1:end-1] # wT[i] for i = 1, ..., Nt-1
    predictions = vavg[2:end] # wT[i+1]
    return (predictors, predictions)
end

function postprocess_prediction(wTi, scaled_prediction, data_processor::wTprocessor)
    # wT[i+1] = G(wT[i])
    return scaled_prediction # do nothing
end

"""
Used in gp.jl.
Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

abstract type Problem end

abstract type SequentialProblem <: Problem end
abstract type ResidualProblem <: Problem end

struct Sequential <: SequentialProblem # for mappings that predict the subsequent timestep from the current timestep
    type::String # "T" or "dT" or "wT"
end

struct Residual <: ResidualProblem # for mappings that predict the true current timestep from a physics-based model's current timestep
    type::String # "KPP" or "TKE"
end

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

struct Residual_KPP <: ResidualProblem
    variable::String
end

struct Residual_TKE <: ResidualProblem
    variable::String
end

"""
get_problem(problem::Problem, data::OceananigansData, timeseries)
----- Description
    Creates an instance of a Problem struct depending on the type of mapping.
----- Arguments
- 'problem': (SequentialProblem).  what mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"))
- 'N²': (Number)                   initial buoyancy stratification
- 'timeseries': (Array)            simulation timeseries [s]
"""
function get_problem(problem::Problem, N², timeseries)

    if typeof(problem) <: SequentialProblem

        if problem.type == "dT"
            Δt = (timeseries[2]-timeseries[1]) / N²
            return Sequential_dT("T", Δt) # Residual_T(variable, Δt)

        elseif problem.type == "T"
            return Sequential_T("T")

        elseif problem.type == "wT"
            return Sequential_wT("wT")

        else; throw(error())
        end

    elseif typeof(problem) <: ResidualProblem

        if problem.type == "KPP"
            return Residual_KPP("T")

        elseif problem.type == "TKE"
            return Residual_TKE("T")

        else; throw(error())
        end

    else; throw(error())
    end

end

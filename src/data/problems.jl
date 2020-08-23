"""
Used in gp.jl.
Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

abstract type Problem end

struct SequentialProblem <: Problem # for mappings from variable[i] --> variable[i+1]
    s::String # "T" or "dT" or "wT"
end

struct ResidualProblem <: Problem
    type::String # "KPP" or "TKE"
end

"""
get_problem(problem::Problem, data::OceananigansData, timeseries)
----- Description
    Creates an instance of a Problem struct depending on whether the variable is T or wT.
----- Arguments
- 'data': (OceananigansData)       struct containing data from simulation.
- 'timeseries': (Array)            simulation timeseries [s]
- 'problem': (SequentialProblem).  what mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"))
"""
function get_problem(problem::SequentialProblem, data::OceananigansData, timeseries)

    α = 2e-4
    g = 9.80665
    # b_initial = 𝒟.T[:,1] .* α*g
    # approximate initial buoyancy gradient N², where b = N²z + 20*α*g
    #                                                 T = N²z/(αg) + 20
    T_initial = data.T[:,1]
    N² = (T_initial[1] - 20)*α*g / data.z[1]

    if problem.type == "dT"
        Δt = (timeseries[2]-timeseries[1]) / N²
        return Residual_T("T", Δt) # Residual_T(variable, Δt)

    elseif problem.type == "T"
        return Sequential_T("T")

    elseif problem.type == "wT"
        return Sequential_wT("wT")

    else
        throw(error())
    end
end


"""
get_problem(problem::Problem, data::OceananigansData, timeseries)
----- Description
    Creates an instance of a Problem struct depending on whether the variable is T or wT.
----- Arguments
- 'problem': (ResidualProblem).    what mapping you wish to evaluate with the model. (Residual("T"), Residual("KPP"), or Residual("TKE"))
- 'data': (OceananigansData)       struct containing data from simulation.
- 'timeseries': (Array)            simulation timeseries [s]
"""
function get_problem(problem::ResidualProblem, data::OceananigansData, timeseries)

    if problem.type == "KPP"
        return Residual_KPP("T")

    elseif problem.type == "TKE"
        return Residual_TKE("T")

    else
        throw(error())
    end
end

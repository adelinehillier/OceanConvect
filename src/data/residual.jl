"""
Used in gp.jl.
Data pre- / post-processing. Takes a ProfileData object and prepares it for use in GP.
"""

include("problems.jl")

struct Residual_KPP <: ResidualProblem
    variable::String
end

struct Residual_TKE <: ResidualProblem
    variable::String
end

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

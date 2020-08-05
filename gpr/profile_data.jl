using LinearAlgebra
using BenchmarkTools

include("kernels.jl")
include("gaussian_process.jl")
include("../les/get_les_data.jl")
include("../les/custom_avg.jl")

"""
ProfileData
# Description
- data structure for preparing profile data from Oceananigans simulations for analysis with gpr, nn, or ed.
# Data Structure and Description
    v::Array,
    x::Array,
    y::Array,
    x_train::Array, training inputs (predictors; array of states). (length-n array of D-length vectors, where D is the length of each input n is the number of training points)
    y_train::Array, training outputs (predictions) (length-n array of D-length vectors).
    x_verification::Array
    y_verification::Array
    z::Vector, depth values averaged to D gridpoints
    t::Array, timeseries [seconds]
    Nt::Int64, length(timeseries)
"""
struct ProfileData
    v
    vavg
    x
    y
    x_train
    y_train
    verification_set
    z
    zavg
    t
    Nt
    n_train
end

"""
construct_profile_data(filename, D; N=4)
# Description
Returns an instance of ProfileData.
# Arguments
- 'filename': (string). Name of the NetCDF or JLD2 file containing the data from the Oceananigans simulation.
- 'D' (integer). Number of gridpoints in the z direction to average the data to for training and prediction.
# Keyword Arguments
- 'N': (integer). Interval between the timesteps to be reserved for training data (default 4).
                If N=4, the profile data for every 4 timesteps will be reserved for training (~25% training data);
                the rest will be used in the verification set.
"""
function construct_profile_data(filename, V_name, D; N=4, verbose=true)

    data = get_les_data(filename)

    # get variable (T or wT) array
    if V_name=="T"
        V=data.T
    elseif V_name=="wT"
        V=data.wT
    else
        throw(error())
    end

    t = data.t
    Nt = length(t)

    total_set = 1:(Nt-1)
    training_set = 1:N:(Nt-1)
    verification_set = setdiff(total_set, training_set)
    n_train=length(training_set)

    # compress variable array to D values per time
    vavg = [custom_avg(V[:,j], D) for j in 1:Nt]
    x = vavg[1:(Nt-1)] # (v₀, v₁, ... ,v_(Nt-1)) (Nt-1)-length array of D-length inputs
    y = vavg[2:(Nt)]   # (v₁, v₂, ... ,v_Nt    ) (Nt-1)-length array of D-length targets
    x_train = x[training_set]
    y_train = y[training_set]
    # x_verification = x[verification_set]
    # y_verification = y[verification_set]

    z = data.z
    zavg = custom_avg(z, D)

    return ProfileData(V, vavg, x, y, x_train, y_train, verification_set, z, zavg, t, Nt, n_train)
end

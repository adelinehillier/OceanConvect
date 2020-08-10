using LinearAlgebra
using BenchmarkTools

include("gaussian_process.jl")
include("scalings.jl")
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
    n_train::Int64, number of training pairs
    κₑ::Float, eddy diffusivity

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
    κₑ
    scaling
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
function construct_profile_data(filename::String, V_name, D; N=4, verbose=true)

    data = get_les_data(filename)

    # get variable (T or wT) array
    if V_name=="T"; V=data.T
    elseif V_name=="wT"; V=data.wT
    else
        throw(error())
    end

    t = data.t
    Nt = length(t)
    κₑ = data.κₑ # eddy diffusivity

    total_set = 1:(Nt-1)
    training_set = 1:N:(Nt-1)
    verification_set = setdiff(total_set, training_set)
    n_train=length(training_set)
    # compress variable array to D values per time
    vavg = [custom_avg(V[:,j], D) for j in 1:Nt]

    scaling = get_scaling(V_name, vavg) # scaling for pre- and post-processing

    #preprocessing
    vavg = [forward(vec, scaling) for vec in vavg]

    x = vavg[1:(Nt-1)] # (v₀, v₁, ... ,v_(Nt-1)) (Nt-1)-length array of D-length inputs
    y = vavg[2:(Nt)]   # (v₁, v₂, ... ,v_Nt    ) (Nt-1)-length array of D-length targets
    x_train = x[training_set]
    y_train = y[training_set]
    # x_verification = x[verification_set]
    # y_verification = y[verification_set]

    # depth
    z = data.z
    zavg = custom_avg(z, D)

    return ProfileData(V, vavg, x, y, x_train, y_train, verification_set, z, zavg, t, Nt, n_train, κₑ, scaling)
end

function construct_profile_data_multiple(filenames::Array, V_name, D; N=4)
    # combines data from multiple files
    𝒟 = construct_profile_data(filenames[1], V_name, D; N=N)

    v = 𝒟.v
    x_train = 𝒟.x_train
    y_train = 𝒟.y_train

    for filename in filenames[2:end]
        data_b = construct_profile_data(filename, V_name, D; N=N)

        training_set = 1:N:(data_b.Nt-1)

        v = hcat(v, data_b.v)
        x_train = vcat(x_train, data_b.x[training_set])
        y_train = vcat(y_train, data_b.y[training_set])
    end

    # ONLY v, x_train and y_train contain data from all filenames, the rest of the attributes are from the first filename in filenames
    return ProfileData(v, 𝒟.vavg, 𝒟.x, 𝒟.y, x_train, y_train, 𝒟.verification_set, 𝒟.z, 𝒟.zavg, 𝒟.t, 𝒟.Nt, 𝒟.n_train, 𝒟.κₑ, 𝒟.scaling)
end

"""
Used in GP1.jl.
ProfileData struct for storing simulation data and functions for creating instances.
"""

using LinearAlgebra
using BenchmarkTools

include("gaussian_process.jl")
include("scalings.jl")
include("pre_post_processing.jl")
include("../les/get_les_data.jl")
include("../les/custom_avg.jl")

"""
ProfileData
------ Description
- data structure for preparing profile data from Oceananigans simulations for analysis with gpr, nn, or ed.
------ Data Structure and Description
    v::Array,           Nz x Nt array of T or wT values directly from the simulation, not preprocessed.
    vavg::Array,        Nt-length array of Nz-length vectors, scaled and pre-processed
    x::Array,           all simulation inputs, scaled and pre-processed
    y::Array,           all simulation inputs, scaled and pre-processed
    x_train::Array,     training inputs (predictors; array of states). (length-n array of D-length vectors, where D is the length of each input n is the number of training points)
    y_train::Array,     training outputs (predictions) (length-n array of D-length vectors).
    verification_set::Array, vector of indices corresponding to verification data
    z::Array,           Nz-length vector of depth values
    zavg::Array,        length-D vector; depth values averaged to D gridpoints
    t::Array,           timeseries [seconds]
    Nt::Int64,          length(timeseries)
    n_train::Int64,     number of training pairs
    κₑ::Float,          eddy diffusivity
    scaling::Scaling,   scaling struct for normalizing the data along the T or wT axis
    processor::DataProcessor, struct for preparing the data for GP regression
"""
struct ProfileData
    v::Array
    vavg::Array
    x::Array
    y::Array
    x_train::Array
    y_train::Array
    verification_set::Array
    z::Array
    zavg::Array
    t::Array
    Nt::Int64
    n_train::Int64
    κₑ::Float64
    scaling::Scaling
    processor::DataProcessor
end

"""
construct_profile_data(filename, D; N=4)
------ Description
Returns an instance of ProfileData.
------ Arguments
- 'filename': (string)  Name of the NetCDF or JLD2 file containing the data from the Oceananigans simulation.
- 'V_name': (string)    "T" for the temperature profile or "wT" for the temperature flux
- 'D' (integer)         Number of gridpoints in the z direction to average the data to for training and prediction.
------ Keyword Arguments
- 'N': (integer)        Interval between the timesteps to be reserved for training data (default 4).
                        If N=4, the profile data for every 4 timesteps will be reserved for training (~25% training data);
                        the rest will be used in the verification set.
"""
function construct_profile_data(filename::String, V_name, D; N=4)

    # collect data from Oceananigans simulation output file
    data = get_les_data(filename)

    # get variable (T or wT) array
    if V_name=="T";
        V=data.T
    elseif V_name=="wT";
        V=data.wT
    else
        throw(error())
    end

    # timeseries [s]
    t = data.t
    Nt = length(t)

    # eddy diffusivity
    κₑ = data.κₑ

    total_set = 1:(Nt-1)
    training_set = 1:N:(Nt-1)
    verification_set = setdiff(total_set, training_set)
    n_train=length(training_set)
    # compress variable array to D values per time
    vavg = [custom_avg(V[:,j], D) for j in 1:Nt]

    # preprocessing
    # 1. scale the data so that it ranges from 0 to 1
    scaling = get_scaling(V_name, vavg)
    vavg = [scale(vec, scaling) for vec in vavg]
    # 2. get the "preprocessed" data for GPR
    processor = get_processor(data, t, V_name)
    x,y = get_predictors_predictions(vavg, processor)
    # x is (v₀, v₁, ... ,v_(Nt-1)) (Nt-1)-length array of D-length inputs
    # y is the predictions for each elt in x

    x_train = x[training_set]
    y_train = y[training_set]
    # x_verification = x[verification_set]
    # y_verification = y[verification_set]

    # depth values
    z = data.z
    zavg = custom_avg(z, D)

    return ProfileData(V, vavg, x, y, x_train, y_train, verification_set, z, zavg, t, Nt, n_train, κₑ, scaling, processor)
end

"""
construct_profile_data(filename, D; N=4)
------ Description
Returns an instance of ProfileData containing training data from multiple simulations.
*** Important:
    ONLY v, x_train and y_train contain data from all filenames;
    the remaining attributes are taken from the first filename in filenames
------ Arguments
- 'filenames': (string)  Vector of filenames (.nc or .jld2) to collect data from.
- 'V_name': (string)     "T" for the temperature profile or "wT" for the temperature flux
- 'D' (integer)          Number of gridpoints in the z direction to average the data to for training and prediction.
------ Keyword Arguments
- 'N': (integer)         Interval between the timesteps to be reserved for training data (default 4).
                         If N=4, the profile data for every 4 timesteps will be reserved for training (~25% training data);
                         the rest will be used in the verification set.
"""
function construct_profile_data_multiple(filenames::Array, V_name, D; N=4)

    # combines data from multiple files
    𝒟 = construct_profile_data(filenames[1], V_name, D; N=N)

    v = 𝒟.v
    x_train = 𝒟.x_train
    y_train = 𝒟.y_train

    for filename in filenames[2:end]
        data_b = construct_profile_data(filename, V_name, D; N=N)

        training_set = 1:N:(data_b.Nt-1)

        v = hcat(v, data_b.v) # unscaled
        x_train = vcat(x_train, data_b.x[training_set])
        y_train = vcat(y_train, data_b.y[training_set])
    end

    # ONLY v, x_train and y_train contain data from all filenames, the rest of the attributes are from the first filename in filenames
    return ProfileData(v, 𝒟.vavg, 𝒟.x, 𝒟.y, x_train, y_train, 𝒟.verification_set, 𝒟.z, 𝒟.zavg, 𝒟.t, 𝒟.Nt, 𝒟.n_train, 𝒟.κₑ, 𝒟.scaling, 𝒟.processor)
end

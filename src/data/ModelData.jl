"""
Data module for preparing data for analysis with

    - GaussianProcess (src/gpr/GaussianProcess.jl)
        or
    - NeuralNetwork (src/gpr/NeuralNetwork.jl)

"""

module ModelData

# using JLD2,
#       NetCDF,
#       BenchmarkTools

include("../les/custom_avg.jl")

# harvesting Oceananigans data
include("../les/get_les_data.jl")
export get_les_data

# normalization
include("scalings.jl")
export  Tscaling,
        wTscaling
export  scale, # normalize
        unscale # un-normalize

# pre- and post-processing on the normalized data
include("problems.jl")
export  Problem,
        Sequential,
        Residual,
        SequentialProblem,
        ResidualProblem,
        get_problem

include("residual.jl")
include("sequential.jl")
export  get_predictors_targets,
        postprocess_prediction

# ProfileData struct
export  ProfileData,
        data

function approx_initial_buoyancy_stratification(temp_array,z)
    α = 2e-4
    g = 9.80665
    T_initial = temp_array[:,1] # b_initial = 𝒟.T[:,1] .* α*g
    N² = (T_initial[1] - 20)*α*g / z[1] # approximate initial buoyancy gradient N², where b = N²z + 20*α*g and T = N²z/(αg) + 20
    return N²
end

"""
ProfileData
------ Description
- data structure for preparing profile data for analysis with gpr or nn.
------ Data Structure and Description
    v::Array,           Nz x Nt array of T or wT values directly from the LES simulation, not preprocessed.
    vavg::Array,        Nt-length array of Nz-length vectors from LES simulation, scaled and pre-processed
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
    problem::Problem,   what mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"), Residual("T"), Residual("KPP"), or Residual("TKE"))

"""
struct ProfileData
    v       ::Array
    vavg    ::Array
    x       ::Array
    y       ::Array
    x_train ::Array
    y_train ::Array
    verification_set::Array
    z       ::Array
    zavg    ::Array
    t       ::Array
    Nt      ::Int64
    n_train ::Int64
    κₑ      ::Float64
    scaling ::Scaling
    problem ::Problem
end

"""
data(filename, problem; D=16, N=4)

------ Description
Returns a ProfileData object based on data from `filename`

------ Arguments
- 'filename': (string)  Name of the NetCDF or JLD2 file containing the data from the Oceananigans simulation.
- 'problem': (Problem). What mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"), Residual("T"), Residual("KPP"), or Residual("TKE"))

------ Keyword Arguments
- 'D' (integer)         Number of gridpoints in the z direction to average the data to for training and prediction.
- 'N': (integer)        Interval between the timesteps to be reserved for training data (default 4).
                        If N=4, the profile data for every 4 timesteps will be reserved for training (~25% training data);
                        the rest will be used in the verification set.
"""
function data(filename::String, problem::Problem; D=16, N=4)

    # collect data from Oceananigans simulation output file
    data = get_les_data(filename) # <: OceananigansData

    # timeseries [s]
    t = data.t
    Nt = length(t)

    # depth values
    z = data.z
    zavg = custom_avg(z, D)

    # approximate buoyancy stratification at the initial timestep
    N² = approx_initial_buoyancy_stratification(data.T,z)

    # problem
    problem = get_problem(problem, N², t)

    # get variable (T or wT) array
    if problem.variable=="T"
        v = data.T # D x Nt array
    elseif problem.variable=="wT"
        v = data.wT # D x Nt array
    else
        throw(error())
    end

    # eddy diffusivity
    κₑ = data.κₑ

    # divide up the data
    total_set = 1:(Nt-1)
    training_set = 1:N:(Nt-1)
    verification_set = setdiff(total_set, training_set)
    n_train = length(training_set)

    # compress variable array to D gridpoints
    vavg = [custom_avg(v[:,j], D) for j in 1:Nt]

    # preprocessing

    # 1) normalize the data so that it ranges from 0 to 1
    scaling = get_scaling(problem.variable, vavg)
    vavg = [scale(vec, scaling) for vec in vavg]

    # 2) get the "preprocessed" data for GPR
    x,y = get_predictors_targets(vavg, problem)
    # x is (v₀, v₁, ... ,v_(Nt-1)) (Nt-1)-length array of D-length inputs
    # y is the predictions for each elt in x

    x_train = x[training_set]
    y_train = y[training_set]

    return ProfileData(v, vavg, x, y, x_train, y_train, verification_set, z, zavg, t, Nt, n_train, κₑ, scaling, problem)
end

"""
data(filename, D; N=4)

------ Description
Returns an instance of ProfileData containing training data from multiple simulations.
*** Important:
    ONLY v, x_train and y_train contain data from all filenames;
    the remaining attributes are taken from the first filename in filenames

------ Arguments
- 'filenames': (string)  Vector of filenames (.nc or .jld2) to collect data from.
- 'problem': (Problem). What mapping you wish to evaluate with the model. (Sequential("T"), Sequential("wT"), Residual("T"), Residual("KPP"), or Residual("TKE"))

------ Keyword Arguments
- 'D' (integer)         Number of gridpoints in the z direction to average the data to for training and prediction.
- 'N': (integer)        Interval between the timesteps to be reserved for training data (default 4).
                        If N=4, the profile data for every 4 timesteps will be reserved for training (~25% training data);
                        the rest will be used in the verification set.
"""
function data(filenames::Array, problem::Problem; D=16, N=4)

    # combines data from multiple files
    𝒟 = data(filenames[1], problem; D=D, N=N)

    v = 𝒟.v
    x_train = 𝒟.x_train
    y_train = 𝒟.y_train

    for filename in filenames[2:end]
        data_b = data(filename, problem; D=D, N=N)

        training_set = 1:N:(data_b.Nt-1)

        v = hcat(v, data_b.v) # unscaled
        x_train = vcat(x_train, data_b.x[training_set])
        y_train = vcat(y_train, data_b.y[training_set])
    end

    # ONLY v, x_train and y_train contain data from all filenames, the rest of the attributes are from the first filename in filenames
    return ProfileData(v, 𝒟.vavg, 𝒟.x, 𝒟.y, x_train, y_train, 𝒟.verification_set, 𝒟.z, 𝒟.zavg, 𝒟.t, 𝒟.Nt, 𝒟.n_train, 𝒟.κₑ, 𝒟.scaling, 𝒟.problem)
end

end # module

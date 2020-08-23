"""
Includes all useful functions for applying GPR to T and wT profiles from Oceananigans.jl simulations.
Uses ProfileData struct to store data and GP struct for performing GPR on the data in ProfileData object.
"""
module OceanConvect

export
        # ModelData / profile_data
        data,
        SequentialProblem,
        ResidualProblem,

        # GaussianProcess / GP
        # construct_gpr,
        uncertainty,
        # model,
        get_gpr_pred,

        # GaussianProcess / kernels
        get_kernel,

        # GaussianProcess / distances
        euclidean_distance,
        derivative_distance,
        antiderivative_distance,

        # GaussianProcess / hyperparameters
        plot_landscapes_compare_error_metrics,
        plot_landscapes_compare_files_me,
        get_min_gamma,
        get_min_gamma_alpha,

        # GaussianProcess / plot_profile
        plot_profile,
        animate_profile

        # kernel options
        #  1   =>   "Squared exponential"         => "Squared exponential kernel:        k(x,x') = σ * exp( - ||x-x'||² / 2γ² )",
        #  2   =>   "Matern 1/2"                  => "Matérn with ʋ=1/2:                 k(x,x') = σ * exp( - ||x-x'|| / γ )",
        #  3   =>   "Matern 3/2"                  => "Matérn with ʋ=3/2:                 k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)",
        #  4   =>   "Matern 5/2"                  => "Matérn with ʋ=5/2:                 k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)",
        #  5   =>   "Rational quadratic w/ α=1"   => "Rational quadratic kernel:         k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",

# modules
using Plots,
      JLD2,
      NetCDF,
      Statistics,
      LinearAlgebra,
      BenchmarkTools

# submodules
include("data/ModelData.jl")
include("gpr/GaussianProcess.jl")

# re-export symbols from submodules
using .ModelData
using .GaussianProcess

end # module

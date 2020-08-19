"""
Includes all useful functions for applying GPR to T and wT profiles from Oceananigans.jl simulations.
Uses ProfileData struct to store data and GP struct for performing GPR on the data.
"""
module GaussianProcess

    using JLD2, Statistics, LinearAlgebra, Plots

    include("gaussian_process.jl")
    export  construct_gpr,
            prediction,
            uncertainty,
            compute_kernel_matrix,
            mean_log_marginal_loss
    export  get_gp,
            get_gpr_pred

    # *--*--*--*--*--*--*--*--*--*--*--*
    # |  harvesting Oceananigans data  |
    # *--*--*--*--*--*--*--*--*--*--*--*

    include("../les/get_les_data.jl")
    export

    # ProfileData struct for storing simulation data and functions for creating instances.
    # data = construct_profile_data(filename, name, D; N=4, verbose=true)
    include("profile_data.jl")
    export construct_profile_data

    # *--*--*--*--*--*--*--*
    # |  kernel functions  |
    # *--*--*--*--*--*--*--*

    include("kernels.jl")
    export  SquaredExponentialI,
            RationalQuadraticI,
            Matern12I,
            Matern32I,
            Matern52I
    export  kernel_function

    # kernel options
    #  1   =>   "Squared exponential"         => "Squared exponential kernel:        k(x,x') = σ * exp( - ||x-x'||² / 2γ² )",
    #  2   =>   "Matern 1/2"                  => "Matérn with ʋ=1/2:                 k(x,x') = σ * exp( - ||x-x'|| / γ )",
    #  3   =>   "Matern 3/2"                  => "Matérn with ʋ=3/2:                 k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)",
    #  4   =>   "Matern 5/2"                  => "Matérn with ʋ=5/2:                 k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)",
    #  5   =>   "Rational quadratic w/ α=1"   => "Rational quadratic kernel:         k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",

    function get_kernel(k::Int64, γ, σ)
        # convert from log10 scale
        γ = 10^γ
        σ = 10^σ
      if k==1; return SquaredExponentialI(γ, σ) end
      if k==2; return Matern12I(γ, σ) end
      if k==3; return Matern32I(γ, σ) end
      if k==4; return Matern52I(γ, σ) end
      if k==5; return RationalQuadraticI(γ, σ, 1.0)
      else; throw(error()) end
    end

    # *--*--*--*--*--*--*--*
    # |  distance metrics  |
    # *--*--*--*--*--*--*--*

    include("distance.jl")
    export  l2_norm,
            h1_norm,
            hm1_norm

    # *--*--*--*--*--*
    # |  scalings    |
    # *--*--*--*--*--*

    include("scalings.jl")
    export  Tscaling,
            wTscaling
    export  scale,
            unscale

    # *--*--*--*--*--*--*--*--*--*--*
    # |  pre- and post-processing   |
    # *--*--*--*--*--*--*--*--*--*--*

    include("pre_post_processing.jl")
    export  get_processor,
            get_predictors_predictions,
            postprocess_prediction,

    # *--*--*--*--*--*--*--*
    # |  calculate errors  |
    # *--*--*--*--*--*--*--*

    include("errors.jl")
    export  get_me_true_check, # evolving forward from an arbitrary initial timestep
            get_me_greedy_check # how well does the mean GP prediction fit the training data?

   # *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
   # |  plot hyperparameter landscapes for optimization    |
   # *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

   export  plot_landscapes_compare_error_metrics,
           plot_landscapes_compare_files_me,
           get_min_gamma,
           get_min_gamma_alpha

end

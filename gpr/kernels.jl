"""
Create constructors for covariance functions.

      Constructor                   Description                                Isotropic/Anisotropic
    - SquaredExponentialI(γ,σ):     squared exponential covariance function    isotropic
    - SquaredExponentialA(γ,σ):     squared exponential covariance function    anisotropic
    - ExponentialI(γ,σ):            exponential covariance function            isotropic
    - ExponentialA(γ,σ):            exponential covariance function            anisotropic
    - RationalQuadraticI():         rational quadratic covariance function     isotropic
    - RationalQuadraticA():         rational quadratic covariance function     anisotropic
    - Matern12I():                  Matérn covariance function with ʋ = 1/2.   isotropic
    - Matern12A():                  Matérn covariance function with ʋ = 1/2.   anisotropic
    - Matern32I():                  Matérn covariance function with ʋ = 3/2.   isotropic
    - Matern32A():                  Matérn covariance function with ʋ = 3/2.   anisotropic
    - Matern52I():                  Matérn covariance function with ʋ = 5/2.   isotropic
    - Matern52A():                  Matérn covariance function with ʋ = 5/2.   anisotropic
"""
abstract type Kernel end

#  *--*--*--*--*--*--*--*--*--*--*--*
#  | Isotropic covariance functions |
#  *--*--*--*--*--*--*--*--*--*--*--*

""" SquaredExponentialI(γ,σ): squared exponential covariance function, isotropic """
struct SquaredExponentialKernelI{T<:Float64} <: Kernel
    # Hyperparameters
    "Squared length scale"
    γ::T
    "Signal variance"
    σ::T
    # equation = "k(a,b) = σ * exp( - ||a-b||^2 / 2*γ )"
    # evaluate(a,b; γ,σ) = σ * exp(- sq_mag(a,b) / 2*γ )
end

function kernel_function(k::SquaredExponentialKernelI)
  evaluate(a,b) = k.σ * exp(- sq_mag(a,b) / k.γ )
  return evaluate
end

""" ExponentialI(γ,σ): exponential covariance function, isotropic """
struct ExponentialKernelI{T<:Float64} <: Kernel
    # Hyperparameters
    "Squared length scale"
    γ::T
    "Signal variance"
    σ::T
    # equation = "k(a,b) = σ * exp( - ||a-b|| / 2*γ )"
    # evaluate(a,b; γ,σ) = σ * exp(- mag(a,b) / 2*γ )
end

function kernel_function(k::ExponentialKernelI)
  evaluate(a,b) = k.σ * exp(- sqrt(sq_mag(a,b)) / k.γ )
  return evaluate
end

struct RationalQuadraticI{T<:Float64} <: Kernel
    # Hyperparameters
    "Squared length scale"
    γ::T
    "Signal variance"
    σ::T
    "Shape parameter"
    α::T
end

function kernel_function(k::RationalQuadraticI)
    function evaluate(a,b)
     return k.σ * (1+(a-b)'*(a-b)/(2*k.α*k.γ))^(-k.α)
 end
  return evaluate
end

struct Matern12I{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    l::T
    "Signal variance"
    σ::T
end

function kernel_function(k::Matern12I)
  evaluate(a,b) = k.σ * exp(- sqrt(sq_mag(a,b)) / k.l )
  return evaluate
end

struct Matern32I{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    l::T
    "Signal variance"
    σ::T
end

function kernel_function(k::Matern32I)
    function evaluate(a,b)
        c = sqrt(3*sq_mag(a,b))/k.l
        return k.σ * (1+c) * exp(-c)
    end
  return evaluate
end

struct Matern52I{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    l::T
    "Signal variance"
    σ::T
end

function kernel_function(k::Matern52I)
    function evaluate(a,b)
        c = sqrt(5*sq_mag(a,b))/k.l
        d = 5*sq_mag(a,b)/(3*k.l^2)
        return k.σ * (1+c+d) * exp(-c)
    end
  return evaluate
end

#  *--*--*--*--*--*--*--*--*--*--*--*--*
#  | Anisotropic covariance functions  |
#  *--*--*--*--*--*--*--*--*--*--*--*--*

# function Matern(ν::Real, ll::Real, lσ::Real)
#     if ν==1/2
#         kern = Mat12Iso(ll, lσ)
#     elseif ν==3/2
#         kern = Mat32Iso(ll, lσ)
#     elseif ν==5/2
#         kern = Mat52Iso(ll, lσ)
#     else throw(ArgumentError("Only Matern 1/2, 3/2 and 5/2 are implementable"))
#     end
#     return kern
# end

# anisotropic_kernel_options = Dict("squared_exponential" => squared_exponential_kernel,
#                       "exponential" => exponential_kernel,
#                     #   "rational_quadratic" => rational_quadratic_kernel,
#                     #   "matern12" => matern_kernel(0.5),
#                     #   "matern32" => matern_kernel(1.5),
#                     #   "matern52" => matern_kernel(2.5),
#                   )

# function kernel_function(kernel_name, hyperparameters)
#     k(a,b) = kernel_options[kernel_name](hyperparameters)
#     return k
# end

# function set_params!(se::Kernel, hyp::AbstractVector)
#     length(hyp) == 2
#     se.ℓ2, se.σ2 = exp(2 * hyp[1]), exp(2 * hyp[2])
# end

# """
# squared_exponential_kernel(x,y; hyperparameters=[1.0, 1.0])
# # Description
# - Outputs a squared exponential kernel with hyperparameters γ, σ
# # Arguments
# - a: first coordinate
# - b: second coordinate
# # Keyword Arguments
# - hyperparameters (vector). = [γ σ]
#   The first is γ, the second is σ where k(a,b) = σ * exp( - ||a-b||^2 / 2*γ )
#     - γ = 1.0: (scalar). squared length-scale
#     - σ = 1.0; (scalar). signal variance
# """
# function squared_exponential_kernel(a,b; hyperparameters=[1.0, 1.0])
#     return hyperparameters[1] * exp(- sq_mag(a,b) / 2*hyperparameters[2] )
# end

# """
# exponential_kernel(x,y; hyperparameters=[1.0, 1.0])
# # Description
# - Outputs a squared exponential kernel with hyperparameters γ, σ
# # Arguments
# - a: first coordinate
# - b: second coordinate
# # Keyword Arguments
# - hyperparameters (vector). = [γ σ]
#   The first is γ, the second is σ where, k(x,y) = σ * exp( - ||a-b|| / 2*γ )
#     - γ = 1.0: (scalar). length-scale
#     - σ = 1.0; (scalar). signal variance
# """
# function exponential_kernel(a,b; hyperparameters=[1.0, 1.0])
#     return hyperparameters[1] * exp(- sqrt(sq_mag(a,b)) / 2*hyperparameters[2] )
# end

# """
# closure_gaussian_kernel(x,y; γ = 1.0, σ = 1.0)
# # Description
# - Outputs a function that computes a Gaussian kernel
# # Arguments
# - d: distance function. d(x,y)
# # Keyword Arguments
# -The first is γ, the second is σ where, k(x,y) = σ * exp(- γ * d(x,y))
# - γ = 1.0: (scalar). hyperparameter in the Gaussian Kernel.
# - σ = 1.0; (scalar). hyperparameter in the Gaussian Kernel.
# """
# function closure_guassian_closure(d; hyperparameters = [1.0, 1.0])
#     function gaussian_kernel(x,y)
#         y = hyperparameters[2] * exp(- hyperparameters[1] * d(x,y))
#         return y
#     end
#     return gaussian_kernel
# end

# function Matern(ν::Real, ll::Real, lσ::Real)
#     if ν==1/2
#         kern = Mat12Iso(ll, lσ)
#     elseif ν==3/2
#         kern = Mat32Iso(ll, lσ)
#     elseif ν==5/2
#         kern = Mat52Iso(ll, lσ)
#     else throw(ArgumentError("Only Matern 1/2, 3/2 and 5/2 are implementable"))
#     end
#     return kern
# end

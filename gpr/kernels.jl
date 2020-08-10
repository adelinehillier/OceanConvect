"""
Constructors for covariance functions.

      Constructor                   Description                                Isotropic/Anisotropic
    - SquaredExponentialI(γ,σ):     squared exponential covariance function    isotropic
  # - SquaredExponentialA(γ,σ):     squared exponential covariance function    anisotropic
    - ExponentialI(γ,σ):            exponential covariance function            isotropic
  # - ExponentialA(γ,σ):            exponential covariance function            anisotropic
    - RationalQuadraticI():         rational quadratic covariance function     isotropic
  # - RationalQuadraticA():         rational quadratic covariance function     anisotropic
    - Matern12I():                  Matérn covariance function with ʋ = 1/2.   isotropic
  # - Matern12A():                  Matérn covariance function with ʋ = 1/2.   anisotropic
    - Matern32I():                  Matérn covariance function with ʋ = 3/2.   isotropic
  # - Matern32A():                  Matérn covariance function with ʋ = 3/2.   anisotropic
    - Matern52I():                  Matérn covariance function with ʋ = 5/2.   isotropic
  # - Matern52A():                  Matérn covariance function with ʋ = 5/2.   anisotropic

Distance metrics

    - l²-norm:  d(x,x') = || x - x' ||",
    - H¹-norm:  d(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||",
    - H⁻¹-norm: d(x,x') = || diff(x).*diff(z) - diff(x').*diff(z) ||"

"""
abstract type Kernel end

#  *--*--*--*--*--*--*--*--*--*--*--*
#  | Isotropic covariance functions |
#  *--*--*--*--*--*--*--*--*--*--*--*

""" SquaredExponentialI(γ,σ): squared exponential covariance function, isotropic """
struct SquaredExponentialI{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
end

# evaluates the kernel function for a given pair of inputs
function kernel_function(k::SquaredExponentialI; d=l2_norm, z=nothing)
    # k(x,x') = σ * exp( - d(x,x')² / 2γ² )
  evaluate(a,b) = k.σ * exp(- d(a,b,z)^2 / 2*(k.γ)^2 )
  return evaluate
end

struct FakeKernelI{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
    # equation = "k(a,b) = σ * exp( - ||a-b||^2 / 2*γ )"
    # evaluate(a,b; γ,σ) = σ * exp(- sq_mag(a,b) / 2*γ )
end

function kernel_function(k::FakeKernelI; d=l2_norm, z=nothing)
  evaluate(a,b,z) = 1.0
  return evaluate
end

struct RationalQuadraticI{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
    "Shape parameter"
    α::T
end

function kernel_function(k::RationalQuadraticI; d=l2_norm, z=nothing)
    # k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)
    function evaluate(a,b)
        l = (k.γ)^2 # squared length scale
     return k.σ * (1+(a-b)'*(a-b)/(2*k.α*l))^(-k.α)
 end
  return evaluate
end

struct Matern12I{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
end

function kernel_function(k::Matern12I; d=l2_norm, z=nothing)
    # k(x,x') = σ * exp( - ||x-x'|| / γ )
  evaluate(a,b) = k.σ * exp(- d(a,b,z) / k.γ )
  return evaluate
end

struct Matern32I{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
end

function kernel_function(k::Matern32I; d=l2_norm, z=nothing)
    # k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)
    function evaluate(a,b)
        c = sqrt(3)*d(a,b,z)/k.γ
        return k.σ * (1+c) * exp(-c)
    end
  return evaluate
end

struct Matern52I{T<:Float64} <: Kernel
    # Hyperparameters
    "Length scale"
    γ::T
    "Signal variance"
    σ::T
end

function kernel_function(k::Matern52I; d=l2_norm, z=nothing)
    # k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)
    function evaluate(a,b)
        g = sqrt(5)*d(a,b,z)/k.γ
        h = 5*(d(a,b,z)^2)/(3*k.γ^2)
        return k.σ * (1+g+h) * exp(-g)
    end
  return evaluate
end

# """ SquaredExponentialI(γ,σ): squared exponential covariance function, isotropic """
# struct SquaredExponentialKernelI{T<:Float64} <: Kernel
#     # Hyperparameters
#     "Length scale"
#     γ::T
#     "Signal variance"
#     σ::T
# end
#
# function kernel_function(k::SquaredExponentialKernelI)
#     # k(x,x') = σ * exp( - ||x-x'||² / 2γ² )
#   evaluate(a,b) = k.σ * exp(- sq_mag(a,b) / 2*(k.γ)^2 )
#   return evaluate
# end
#
# struct FakeKernelI{T<:Float64} <: Kernel
#     # Hyperparameters
#     "Length scale"
#     γ::T
#     "Signal variance"
#     σ::T
#     # equation = "k(a,b) = σ * exp( - ||a-b||^2 / 2*γ )"
#     # evaluate(a,b; γ,σ) = σ * exp(- sq_mag(a,b) / 2*γ )
# end
#
# function kernel_function(k::FakeKernelI)
#   evaluate(a,b) = 1.0
#   return evaluate
# end
#
# struct RationalQuadraticI{T<:Float64} <: Kernel
#     # Hyperparameters
#     "Length scale"
#     γ::T
#     "Signal variance"
#     σ::T
#     "Shape parameter"
#     α::T
# end
#
# function kernel_function(k::RationalQuadraticI)
#     # k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)
#     function evaluate(a,b)
#         l = (k.γ)^2 # squared length scale
#      return k.σ * (1+(a-b)'*(a-b)/(2*k.α*l))^(-k.α)
#  end
#   return evaluate
# end
#
# struct Matern12I{T<:Float64} <: Kernel
#     # Hyperparameters
#     "Length scale"
#     γ::T
#     "Signal variance"
#     σ::T
# end
#
# function kernel_function(k::Matern12I)
#     # k(x,x') = σ * exp( - ||x-x'|| / γ )
#   evaluate(a,b) = k.σ * exp(- mag(a,b) / k.γ )
#   return evaluate
# end
#
# struct Matern32I{T<:Float64} <: Kernel
#     # Hyperparameters
#     "Length scale"
#     γ::T
#     "Signal variance"
#     σ::T
# end
#
# function kernel_function(k::Matern32I)
#     # k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)
#     function evaluate(a,b)
#         c = sqrt(3)*mag(a,b)/k.γ
#         return k.σ * (1+c) * exp(-c)
#     end
#   return evaluate
# end
#
# struct Matern52I{T<:Float64} <: Kernel
#     # Hyperparameters
#     "Length scale"
#     γ::T
#     "Signal variance"
#     σ::T
# end
#
# function kernel_function(k::Matern52I)
#     # k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)
#     function evaluate(a,b)
#         c = sqrt(5)*mag(a,b)/k.γ
#         d = 5*sq_mag(a,b)/(3*k.γ^2)
#         return k.σ * (1+c+d) * exp(-c)
#     end
#   return evaluate
# end

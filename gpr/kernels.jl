"""
Create constructors for covariance functions.

      Constructor                    Description                                Isotropic/Anisotropic
    - SquaredExponentialIso(γ,σ):    squared exponential covariance function    isotropic
    - SquaredExponentialAniso(γ,σ):  squared exponential covariance function    anisotropic
    - ExponentialIso(γ,σ):           exponential covariance function            isotropic
    - ExponentialAniso(γ,σ):         exponential covariance function            anisotropic
    - RationalQuadraticIso():        rational quadratic covariance function     isotropic
    - Matern12Iso():                 Matérn covariance function with ʋ = 1/2.   isotropic
    - Matern12Aniso():               Matérn covariance function with ʋ = 1/2.   anisotropic
    - Matern32Iso():                 Matérn covariance function with ʋ = 3/2.   isotropic
    - Matern32Aniso():               Matérn covariance function with ʋ = 3/2.   anisotropic
    - Matern52Iso():                 Matérn covariance function with ʋ = 5/2.   isotropic
    - Matern52Aniso():               Matérn covariance function with ʋ = 5/2.   anisotropic
"""
abstract type Kernel end

#  *--*--*--*--*--*--*--*--*--*--*--*
#  | Isotropic covariance functions |
#  *--*--*--*--*--*--*--*--*--*--*--*

struct SquaredExponentialKernelIso{T<:Real} <: Kernel
    # Hyperparameters
    "Squared length scale"
    γ::T
    "Signal variance"
    σ::T
    # equation = "k(a,b) = σ * exp( - ||a-b||^2 / 2*γ )"
    # evaluate(a,b; γ,σ) = σ * exp(- sq_mag(a,b) / 2*γ )
end

function kernel_function(k::SquaredExponentialKernelIso)
  evaluate(a,b) = k.σ * exp(- sq_mag(a,b) / 2*k.γ )
  return evaluate
end


#  *--*--*--*--*--*--*--*--*--*--*--*--*
#  | Anisotropic covariance functions  |
#  *--*--*--*--*--*--*--*--*--*--*--*--*

function Matern(ν::Real, ll::Real, lσ::Real)
    if ν==1/2
        kern = Mat12Iso(ll, lσ)
    elseif ν==3/2
        kern = Mat32Iso(ll, lσ)
    elseif ν==5/2
        kern = Mat52Iso(ll, lσ)
    else throw(ArgumentError("Only Matern 1/2, 3/2 and 5/2 are implementable"))
    end
    return kern
end

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


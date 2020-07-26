"""
Adapted from sandreza/Learning/sandbox/gaussian_process.jl
https://github.com/sandreza/Learning/blob/master/sandbox/gaussian_process.jl
Changed handling of kernel functions; changed some variable names
"""

using LinearAlgebra
using BenchmarkTools

include("kernels.jl")
include("../les/custom_avg.jl")

"""
GP
# Description
- data structure for typical GPR computations
# Data Structure and Description
    kernel::ℱ, a function
    data::𝒮 , an array of vectors
    α::𝒮2 , an array
    K::𝒰 , matrix or sparse matrix
    CK::𝒱, cholesky factorization of K
"""
struct GP{ℱ, 𝒮, 𝒮2, 𝒰, 𝒱}
    kernel::ℱ
    data::𝒮
    α::𝒮2
    K::𝒰
    CK::𝒱
end

"""
construct_gpr(x_train, y_train; kernel; sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))
# Description
Constructs the posterior distribution for a GP. In other words this does the 'training' automagically.
# Arguments
- 'x_train': (array). training inputs (predictors), must be an array of states.
                     array is D x n, where D is the length of each input n is the number of training points
- 'y_train': (array). training outputs (prediction), must have the same number as x_train
                     array is 1 x n
- 'kernel': (Kernel). Kernel object. See kernels.jl for options.
                      kernel_function(kernel)(x,x') maps predictor x predictor to real numbers.
# Keyword Arguments
- 'hyperparameters': (array). default = []. hyperparameters that enter into the kernel
- 'sparsity_threshold': (number). default = 0.0. a number between 0 and 1 that determines when to use sparse array format. The default is to never use it
- 'robust': (bool). default = true. This decides whether to uniformly scale the diagonal entries of the Kernel Matrix. This sometimes helps with Cholesky factorizations.
- 'entry_threshold': (number). default = sqrt(eps(1.0)). This decides whether an entry is "significant" or not. For typical machines this number will be about 10^(-8) * largest entry of kernel matrix.
# Return
- 'GP Object': (GP).
"""
function construct_gpr(x_train, y_train, kernel::Kernel; sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))

    # get k(x,x') function from kernel object
    kernel = kernel_function(kernel)

    # fill kernel matrix with values
    K = compute_kernel_matrix(kernel, x_train)

    # get the maximum entry for scaling and sparsity checking
    mK = maximum(K)

    # make Cholesky factorization work by adding a small amount to the diagonal
    if robust
        K += mK*sqrt(eps(1.0))*I
    end

    # check sparsity
    bools = K .> entry_threshold * mK
    sparsity = sum(bools) / length(bools)
    if sparsity < sparsity_threshold
        sparse_K = similar(K) .* 0
        sparse_K[bools] = sK[bools]
        K = sparse(Symmetric(sparse_K))
        CK = cholesky(K)
    else
        CK = cholesky(K)
    end

    # get prediction weights FIX THIS SO THAT IT ALWAYS WORKS
    y = hcat(y_train...)'
    α = CK \ y # α = K + σ_noise*I

    # construct struct
    return GP(kernel, x_train, α, K, CK)
end

"""
prediction(x, 𝒢::GP)
# Description
- Given state x and GP 𝒢, make a prediction
# Arguments
- 'x': state
# Return
- 'y': prediction
"""
function prediction(x, 𝒢::GP)
    println("𝒢.data $(𝒢.data)") #x_train
    println("x $(x)") #x
    y =  𝒢.α' * 𝒢.kernel.(x, 𝒢.data)
    return y
end

"""
uncertainty(x, 𝒢::GP)
# Description
- Given state x and GP 𝒢, output the variance at a point
# Arguments
- 'x': state
# Return
- 'var': variance
"""
function uncertainty(x, 𝒢::GP)
    tmpv = zeros(size(𝒢.data)[1])
    for i in eachindex(𝒢.data)
        tmpv[i] = 𝒢.kernel(x, 𝒢.data[i])
    end
    # no ldiv for suitesparse
    tmpv2 = 𝒢.CK \ tmpv
    var = k(x, x) .- tmpv'*tmpv2  # var(f*) = k(x*,x*) - tmpv'*tmpv2
    return var
end

"""
compute_kernel_matrix(kernel, x)
# Description
- Computes the kernel matrix for GPR
# Arguments
- k : (Kernel) kernel function k(a,b).
- x : (array of predictors). x[1] is a vector
# Return
- sK: (symmetric matrix). A symmetric matrix with entries sK[i,j] = k(x[i], x[j]). This is only meaningful if k(x,y) = k(y,x) (it should)
"""
function compute_kernel_matrix(k, x)

    K = [k(x[i], x[j]) for i in eachindex(x), j in eachindex(x)]

    if typeof(K[1,1]) <: Number
        sK = Symmetric(K)
    else
        sK = K
    end
    return sK
end

# function kernel_function(kernel_name, hyperparameters)
#     k(a,b) = isotropic_kernel_options[kernel_name](hyperparameters)
#     return k
# end

# this is norm(a-b)^2 but more efficient
function sq_mag(a,b) # ||a - b||^2
    ll = 0.0
    indices = 1:length(a)
    @inbounds for k in indices
        ll += (a[k]-b[k])^2
    end
    return ll
end

"""
Adapted from sandreza/Learning/sandbox/gaussian_process.jl
https://github.com/sandreza/Learning/blob/master/sandbox/gaussian_process.jl
Changed handling of kernel functions; changed some variable names;
added log marginal likelihood function.
"""
# previous
using LinearAlgebra
# include("../data/problems.jl")
include("kernels.jl") # covariance functions
# ---

# using .ModelData
# include("../data/ModelData.jl")
# using OceanConvect.ModelData
# import OceanConvect.ModelData

"""
GP
# Description
- data structure for typical GPR computations
# Data Structure and Description
    kernel::ℱ, a Kernel object
    data::𝒮 , an array of vectors (n-length array of D-length vectors)
    α::𝒮2 , an array
    K::𝒰 , matrix or sparse matrix
    CK::𝒱, cholesky factorization of K
    nc::Number, normalization constant (scales the data during preprocessing. The scaling is reversed during postprocessing.)
    d:Function, distance function to use in the kernel
    z::Vector, values w.r.t. which to derivate the state when evaluating the distance function
"""
struct GP{Kernel, 𝒮, 𝒮2, 𝒰, 𝒱}
    kernel::Kernel
    data::𝒮
    α::𝒮2
    K::𝒰
    CK::𝒱
end

"""
construct_gpr(x_train, y_train; kernel; sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))
# Description
Constructs the posterior distribution for a gp. In other words this does the 'training' automagically.
# Arguments
- 'x_train': (array). training inputs (predictors), must be an array of states.
                      length-n array of D-length vectors, where D is the length of each input n is the number of training points.
- 'y_train': (array). training outputs (prediction), must have the same number as x_train
                      length-n array of D-length vectors.
- 'kernel': (Kernel). Kernel object. See kernels.jl.
                      kernel_function(kernel)(x,x') maps predictor x predictor to real numbers.
# Keyword Arguments
- 'z': (vector). values w.r.t. which to derivate the state (default none).
- 'normalize': (bool). whether to normalize the data during preprocessing and reverse the scaling for postprocessing. Can lead to better performance.
- 'hyperparameters': (array). default = []. hyperparameters that enter into the kernel
- 'sparsity_threshold': (number). default = 0.0. a number between 0 and 1 that determines when to use sparse array format. The default is to never use it
- 'robust': (bool). default = true. This decides whether to uniformly scale the diagonal entries of the Kernel Matrix. This sometimes helps with Cholesky factorizations.
- 'entry_threshold': (number). default = sqrt(eps(1.0)). This decides whether an entry is "significant" or not. For typical machines this number will be about 10^(-8) * largest entry of kernel matrix.
# Return
- GP object
"""
function model(x_train, y_train, kernel::Kernel, zavg; sparsity_threshold = 0.0, robust = true, entry_threshold = sqrt(eps(1.0)))

    # all data preprocessing occurs elsewhere.

    # get k(x,x') function from kernel object
    kernel = kernel_function(kernel; z=zavg)
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
    return GP(kernel, x_train, α, K, Array(CK))
end

function model(data::ProfileData; kernel::Kernel = Kernel())
    # create instance of GP using data from ProfileData object
    𝒢 = model(data.x_train, data.y_train, kernel, data.zavg);
    return 𝒢
end

"""
prediction(x, 𝒢::GP)
# Description
- Given scaled state x, GP 𝒢, returns a scaled prediction
# Arguments
- 'x': scaled state
- '𝒢': GP object with which to make the prediction
# Return
- 'y': scaled prediction
"""
function prediction(x, 𝒢::GP)
    return 𝒢.α' * 𝒢.kernel.([x], 𝒢.data)
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

function mean_log_marginal_loss(y_train, 𝒢::GP; add_constant=false)
    """
    Computes log marginal loss for each element in the output and averages the results.
    Assumes noise-free observations.

    log(p(y|X)) = -(1/2) * (y'*α + 2*sum(Diagonal(CK)) + n*log(2*pi))
    where n is the number of training points and

    # Arguments
    - 'y_train': (Array). training outputs (prediction), must have the same number as x_train
    """
    n = length(𝒢.data)
    D = length(𝒢.data[1])

    ys = hcat(y_train...)' # n x D

    if add_constant
        c = sum([log(𝒢.CK[i,i]) for i in 1:n]) + 0.5*n*log(2*pi)
        total_loss=0.0
        for i in 1:D
            total_loss -= 0.5*ys[:,i]'*𝒢.α[:,i] + c
        end
    else
        total_loss=0.0
        for i in 1:D
            total_loss -= 0.5*ys[:,i]'*𝒢.α[:,i]
        end
    end

    return total_loss / D
end

"""
----- Description
Predict profile across all time steps for the true check.
    - if the problem is sequential, predict profiles from start to finish without the training data, using only the initial profile
    - if the problem is residual, predict profiles at each timestep using model-predicted difference between truth and physics-based model (KPP or TKE) prediction

Returns an n-length array of D-length vectors, where n is the number of training points and D is the
----- Arguments
- '𝒢' (GP). The GP object
- '𝒟' (ProfileData). The ProfileData object whose starting profile will be evolved forward using 𝒢.
----- Keyword Arguments
- 'unscaled' (bool). If true, unscale the data for plotting (false for calculating loss).
"""

function get_gpr_pred(𝒢::GP, 𝒟::ProfileData; unscaled=true)

    if typeof(𝒟.problem) <: SequentialProblem

        # Predict temperature profile from start to finish without the training data.
        gpr_prediction = similar(𝒟.vavg[1:𝒟.Nt-1])
        starting = 𝒟.x[1] # x is scaled
        gpr_prediction[1] = starting
        for i in 1:(𝒟.Nt-2)
            x = gpr_prediction[i]
            scaled_model_output = prediction(x, 𝒢)
            gpr_prediction[i+1] = postprocess_prediction(x, scaled_model_output, 𝒟.problem)
        end

    elseif typeof(𝒟.problem) <: ResidualProblem

        # Predict temperature profile at each timestep using model-predicted difference between truth and physics-based model (KPP or TKE) prediction
        gpr_prediction = similar(𝒟.vavg[1:𝒟.Nt-1])
        for i in 1:(𝒟.Nt-1)
            x = gpr_prediction[i]
            scaled_model_output = prediction(x, 𝒢)
            gpr_prediction[i] = postprocess_prediction(x, scaled_model_output, 𝒟.problem)
        end

    else; throw(error)

    end

    if unscaled
        # post-processing - unscale the data for plotting (false for calculating loss)
        gpr_prediction = [unscale(vec, 𝒟.scaling) for vec in gpr_prediction]
    end
    return gpr_prediction
end

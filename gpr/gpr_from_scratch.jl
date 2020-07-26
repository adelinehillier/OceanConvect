"""
Implementing the algorithm in C. E. Rasmussen & C. K. I. Williams, Gaussian
Processes for Machine Learning ch.2, pp.19. to verify the result from
gaussian_process.jl.
"""

using JLD2, NetCDF, Statistics, LinearAlgebra, Plots
include("kernels.jl")
include("gaussian_process.jl")
include("../les/get_les_data.jl")
include("../les/custom_avg.jl")

filename = "general_strat_16_profiles.jld2"
data = get_les_data(filename);

V = data.T
t = data.t
Nz, Nt = size(V)

D = 16 # gridpoints
zavg = custom_avg(data.z, D) # compress z vector to D values
vavg = [custom_avg(V[:,j], D) for j in 1:Nt] # compress variable array to D values per time
x = vavg[1:(Nt-1)] # (v₀, v₁, ... ,v_(Nt-1)) (Nt-1)-length array of D-length inputs
y = vavg[2:Nt]     # (v₁, v₂, ... ,v_Nt    ) (Nt-1)-length array of D-length targets

# reserve 25% of data for training, but across the entire time interval
total_set = 1:(Nt-1)
training_set = 1:4:(Nt-1)
verification_set = setdiff(total_set, training_set)

n = length(training_set) # num. training points
x_train = x[training_set]
y_train = y[training_set]

kernel = SquaredExponentialKernelI(1000.0,1.0) #[γ,σ]


#### From scratch - to predict the whole temperature profile, so D=16 independent GPs

k = kernel_function(kernel)

K = [k(x[i], x[j]) for i in eachindex(x), j in eachindex(x)] # n x n

CK = cholesky(K + maximum(K)*sqrt(eps(1.0))*I) # n x n
CK = Array(CK) # n x n

y = hcat(y_train...)' # n x D

# α_s = CK' \ (CK \ y_s)
α = (CK \ y)

# pick a test point that's not in the training points
x_star = x[100] # ℝ^D

# vector of covariances between x_star and the training points
k_star = [k(x_star,x_train[i]) for i in 1:n] # ℝ^n

y_prediction = k_star' * α # ℝ^D

# using gaussian_process.jl
𝒢 = construct_gpr(x_train, y_train, kernel)
y_prediction_s = prediction([x_star], 𝒢)

println(y_prediction)
println(y_prediction_s)

#### BELOW - same code, but just for one GP instead of 16 -- predicting just one value (the fourth) in the temperature profile

# get k(x,x') function from kernel object
k = kernel_function(kernel)

# fill kernel matrix with values
K = [k(x[i], x[j]) for i in eachindex(x), j in eachindex(x)] # n x n

# make Cholesky factorization work by adding a small amount to the diagonal
CK = cholesky(K + maximum(K)*sqrt(eps(1.0))*I) # n x n

y_s = hcat(y_train...)' # n x D

# predict just fourth point
y = y_s[:,4] # ℝ^n

CK = Array(CK) # n x n

# α_s = CK' \ (CK \ y_s)
# α = CK' \ (CK \ y)
α = (CK \ y) # ℝ^n

# pick a test point that's not in the training points
x_star = x[100] # ℝ^D

# vector of covariances between x_star and training points in x_train
k_star = zeros(n) # ℝ^n
for i in 1:n
    k_star[i]=k(x_star,x_train[i])
end

y_prediction = k_star' * α # ℝ

#----using gaussian_process.jl

𝒢 = construct_gpr(x_train, y_train, kernel)

y_prediction_s = prediction([x_star], 𝒢)[4]

y_prediction == y_prediction_s # true ✓

### using GaussianProcesses.jl package

using GaussianProcesses

x_validation = x[verification_set]
xs_verification = Array(hcat(x_validation...))

xs = Array(hcat(x_train...)) # n x D
ys = Array(hcat(y_train...)) # n x D
gp = GPE(xs,ys[4,:],MeanZero(),SE(20,20) )
μ, variance = predict_f(gp,xs_verification);

x_all = Array(hcat(x_validation...))
μ_all, variance_all = predict_f(gp,x_all);
y_prediction_ss = μ_all[100] # 19.21918518190362

# Optimise the hyperparameters of Gaussian process gp based on type II maximum likelihood estimation
optimize!(gp)

# mcmc(gp; kern=true)

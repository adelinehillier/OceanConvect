"""
Adapted from sandreza/Learning/sandbox/learn_simple_convection.jl
https://github.com/sandreza/Learning/blob/master/sandbox/learn_simple_convection.jl
"""

using Statistics, LinearAlgebra, Plots
include("kernels.jl")
include("gaussian_process.jl")
include("../les/get_les_data.jl")

const save_figure = false

filename = "general_strat_16_profiles.jld2"
data = get_les_data(filename);

# pick variable to model
V = data.T;
# V = data.wT

t = data.t;
Nz, Nt = size(V);

D = 16 # gridpoints
zavg = custom_avg(data.z, D); # compress z vector to D values
vavg = [custom_avg(V[:,j], D) for j in 1:Nt]; # compress variable array to D values per time
x = vavg[1:(Nt-1)]; # (v₀, v₁, ... ,v_(Nt-1)) (Nt-1)-length array of D-length inputs
y = vavg[2:Nt];     # (v₁, v₂, ... ,v_Nt    ) (Nt-1)-length array of D-length targets

# reserve 25% of data for training, but across the entire time interval
total_set = 1:(Nt-1);
training_set = 1:4:(Nt-1);
verification_set = setdiff(total_set, training_set);

# n_train = length(training_set)
x_train = x[training_set];
y_train = y[training_set];

kernel = SquaredExponentialKernelIso(1.0,1.0); #[γ,σ]
𝒢 = construct_gpr(x_train, y_train, kernel);

index_check = 1
y_prediction = prediction([x_train[index_check]], 𝒢)
norm(y_prediction - y_train[index_check])

# get error at each time in the verification set
gpr_error = collect(verification_set)*1.0;
# greedy check
for j in eachindex(verification_set);
    test_index = verification_set[j];
    y_prediction = prediction([x[test_index]], 𝒢);
    δ = norm(y_prediction - y[test_index]);
    gpr_error[j] = δ;
end
histogram(gpr_error)

#error across all verification time steps
println("The mean error is " * string(sum(gpr_error)/length(gpr_error)))
println("The maximum error is " * string(maximum(gpr_error)))
# the true check
# time evolution given the same initial condition
Nt = length(data.t)
set = 1:(Nt-1)
gpr_prediction = similar(y[total_set])
starting = x[1]
gpr_prediction[1] = starting
Nt = length(y[total_set])
for i in set
    gpr_prediction[i+1] = prediction([gpr_prediction[i]], 𝒢)
end

animation_set = 1:30:(Nt-1)
anim = @animate for i in animation_set
    exact = V[:,i+1]
    day_string = string(floor(Int, data.t[i]/86400))
    p1 = scatter(gpr_prediction[i+1], zavg, label = "GP")
    plot!(exact,data.z, legend = :topleft, label = "LES", xlabel = "temperature", ylabel = "depth", title = "day " * day_string, xlims = (19,20))
    display(p1)
end
if save_figure == true
    gif(anim, pwd() * "gp_emulator.gif", fps = 15)
    mp4(anim, pwd() * "gp_emulator.mp4", fps = 15)
end

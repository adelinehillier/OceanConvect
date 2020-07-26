"""
Adapted from sandreza/Learning/sandbox/learn_simple_convection.jl
https://github.com/sandreza/Learning/blob/master/sandbox/learn_simple_convection.jl
"""

using JLD2, Statistics, LinearAlgebra, Plots, Interact

const save_figure = false

file = togglebuttons(Dict("general_strat_8_profiles"=>"general_strat_8_profiles/general_strat_8_profiles.jld2",
            "general_strat_16_profiles"=>"general_strat_16_profiles/general_strat_16_profiles.jld2",
            "general_strat_32_profiles"=>"general_strat_32_profiles/general_strat_32_profiles.jld2"))

filename = pwd()*"/les_data_sandreza/"*file[]

include(pwd()*"/les/custom_functions.jl")
include(pwd()*"/gpr/gaussian_process.jl")
if filename[end-3:end]==".nc"; include(pwd()*"/les/convert_netcdf_to_data.jl")
else include(pwd()*"/les/convert_jld2_to_data.jl") end
data = OceananigansData(filename)

t = data.t
Nz, Nt = size(data.T)
gp = 16 # gridpoints
zavg = avg(data.z, gp) #avg z vector down to gp values

var = togglebuttons(Dict("T"=>data.T, "wT"=>data.wT))
vavg = [avg(var[][:,j], gp) for j in 1:Nt] #avg variable array down to gp values per time
x = vavg[1:(Nt-1)] # (v₀ v₁ ... v_(Nt-1))
y = vavg[2:Nt]     # (v₁ v₂ ... v_Nt) targets
# ∂t(T) = - ∂z(wT) +  ∂z(∂z(T))
# T(n+1) = T(n) + Δt * (- ∂z(wT) +  ∂z(∂z(T)))
# Gp = -∂z(wT)
# y = T(n+1) - T(n) - Δt * ∂z(∂z(T))

Nt = length(t)
total_set = 1:(Nt-1)
training_set = 1:4:(Nt-1) # 25% of the data, but the entire interval
verification_set = setdiff(total_set, training_set)

x_train = x[training_set]
y_train = y[training_set]

# these are the hyperparameter knobs
σ1 = slider(0:0.1:2, label="σ1")
γ1 = slider(0:0.1:2, label="σ1")

# RBF_plot =
# squared_exponential_plot =

kernel_choice = tabulator(OrderedDict("RBF"=>k(a,b) = σ1[] * exp(- γ1[] * distance(a,b)),
                                    "Hello"=>k(a,b) = σ1[] * exp(- γ1[] * (a-b)),

                                    ))

k(a,b) = σ1[] * exp(- γ1[] * distance(a,b))

#fill kernel matrix with values
n=100
tval = collect(1:n)
kmat = [k(tval[i],tval[j]) for i in 1:n, j in 1:n] #array comprehension!
k_plot = heatmap(kmat, title = "Correlation Matrix")


# d(a,b) = norm(a-b)^2
# cc = closure_guassian_closure(d, hyperparameters = [γ1[],σ1[]])
𝒢 = construct_gpr(x_train, y_train, k);

# index_check = 1
# y_prediction = prediction([x_train[index_check]], 𝒢)
# norm(y_prediction - y_train[index_check])

gpr_error = collect(verification_set)*1.0
# greedy check
for j in eachindex(verification_set)
    test_index = verification_set[j]
    y_prediction = prediction([x[test_index]], 𝒢)
    δ = norm(y_prediction - y[test_index])
    gpr_error[j] = δ
end

mean_error = string(sum(gpr_error)/length(gpr_error))
# println("The mean error is " * string(sum(gpr_error)/length(gpr_error))) #error across all time steps
# println("The maximum error is " * string(maximum(gpr_error)))
error_plot = histogram(gpr_error, xlabel="Error", ylabel="Frequency",xlims=(0.0,2e-5),ylims=(0,500), label="frequency")
vline!([1e-5], line = (4, :dash, 0.6), label="mean error")

# the true check
# time evolution given the same initial condition
Nt = length(data.t)
set = 1:(Nt-1)
gpr_prediction = similar(y[total_set])
starting = x[1]
gpr_prediction[1] = starting
Nt = length(y[set])
for i in set
    gpr_prediction[i+1] = prediction([gpr_prediction[i]], 𝒢)
end

l = @layout [a b; c d]
# p1 = plot(gpr_error)
# p3 = histogram(gpr_error)
# p4 = histogram(gpr_error)
# plot(p1,p2,p3,p4, layout = l)

time = slider(1:100:Nt-5)

exact = data.T[:,time[]+1]
day_string = string(floor(Int, data.t[time[]]/86400))
p1 = scatter(gpr_prediction[time[]+1], zavg, label = "GP")
plot!(exact,data.z, legend = :topleft, label = "LES", xlabel = "temperature", ylabel = "depth", title = "day " * day_string, xlims = (19,20))
display(p1)

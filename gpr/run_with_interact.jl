"""
Adapted from sandreza/Learning/sandbox/learn_simple_convection.jl
https://github.com/sandreza/Learning/blob/master/sandbox/learn_simple_convection.jl
"""

using JLD2, Statistics, LinearAlgebra, Plots, Interact
using Mux, WebIO, Blink

include("kernels.jl")
include("gaussian_process.jl")
include("../les/get_les_data.jl")
const save_figure = false

# function app(req)
  filename = togglebuttons(Dict("general_strat_8_profiles" =>"general_strat_8_profiles.jld2",
                                "general_strat_16_profiles"=>"general_strat_16_profiles.jld2",
                                "general_strat_32_profiles"=>"general_strat_32_profiles.jld2"))
  data = get_les_data(filename[])


  # pick variable to model
  V = data.T;
  # V = data.wT

  t = data.t;
  Nz, Nt = size(V);

  D = 16 # gridpoints
  zavg = custom_avg(data.z, D); # compress z vector to D values

  V = togglebuttons(Dict("T"=>data.T, "wT"=>data.wT))

  vavg = [custom_avg(V[][:,j], D) for j in 1:Nt]; # compress variable array to D values per time
  x = vavg[1:(Nt-1)]; # (v₀, v₁, ... ,v_(Nt-1)) (Nt-1)-length array of D-length inputs
  y = vavg[2:Nt];     # (v₁, v₂, ... ,v_Nt    ) (Nt-1)-length array of D-length targets

  x_train = x[training_set]
  y_train = y[training_set]

  # these are the hyperparameter knobs on a log scale (# log(σ)=a -> σ=10*a)
  σ1 = slider(0:0.1:2, label="log signal variance, log(σ)")
  γ1 = slider(0:0.1:2, label="log length scale,    log(γ)")
  σ = 10^σ1[]
  γ = 10^γ1[]
  kern = tabulator(Dict("Squared exponential"          => SquaredExponentialKernelI(σ,γ),
                                 "Exponential"         => ExponentialKernelI(σ,γ),
                               # "Rational quadratic"  => RationalQuadraticKernelI(σ,γ),
                                 "Matern 1/2"          => Matern12I(σ,γ),
                                 "Matern 3/2"          => Matern32I(σ,γ),
                                 "Matern 5/2"          => Matern52I(σ,γ)
                                 ))

  kern = tabulator(Dict("Squared exponential"          => 5,
                                "Exponential"          => 2,
                              # "Rational quadratic"   => 3,
                                "Matern 1/2"           => 4,
                                "Matern 3/2"           => 1,
                                "Matern 5/2"           => 6
                                ))

  ks = Dict(6                      => SquaredExponentialKernelI(σ,γ),
                                 2         => ExponentialKernelI(σ,γ),
                               # "Rational quadratic"  => RationalQuadraticKernelI(σ,γ),
                                 4          => Matern12I(σ,γ),
                                 1          => Matern32I(σ,γ),
                                 6          => Matern52I(σ,γ)
                                 )

 hyper_landscape = plot([1 2 3],[3 2 1])
 scatter!([σ1[], γ1[]])

 options = hbox(filename,V)
 kernel_plots = hbox(k_plot, hyper_landscape)

  kernel=ks[kern[]]

  #fill kernel matrix with values
  n=100
  kmat = [kernel_function(kernel)(i,j) for i in 1:n, j in 1:n] #array comprehension!
  k_plot = heatmap(kmat, title = "Covariance Matrix", xaxis=(:false), yaxis=(:flip, :false), legend=false)

  𝒢 = construct_gpr(x_train, y_train, kernel);

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

  mean_error = sum(gpr_error)/length(gpr_error)
  # println("The mean error is " * string(sum(gpr_error)/length(gpr_error))) #error across all time steps
  # println("The maximum error is " * string(maximum(gpr_error)))
  error_plot_log = histogram(log.(gpr_error), title = "log(Error)", xlabel="log(Error)", ylabel="Frequency",xlims=(-20,0.0),ylims=(0,200), label="frequency")
  vline!([log(mean_error)], line = (4, :dash, 0.6), label="mean error")

  error_plot = histogram(gpr_error, title = "log(Error)", xlabel="Error", ylabel="Frequency",xlims=(0.0,3e-5),ylims=(0,500), label="frequency")
  vline!([mean_error], line = (4, :dash, 0.6), label="mean error")

  # the true check
  # time evolution given the same initial condition
  Nt = length(data.t)
  gpr_prediction = similar(y[total_set])
  starting = x[1]
  gpr_prediction[1] = starting
  Nt = length(y[total_set])
  for i in 1:(Nt-2)
      gpr_prediction[i+1] = prediction([gpr_prediction[i]], 𝒢)
  end

  l = @layout [a b; c d]
  # p1 = plot(gpr_error)
  # p3 = histogram(gpr_error)
  # p4 = histogram(gpr_error)
  # plot(p1,p2,p3,p4, layout = l)

  time = slider(1:100:Nt-5, label="time step")

  exact = V[][:,time[]+1]
  day_string = string(floor(Int, data.t[time[]]/86400))
  p1 = scatter(gpr_prediction[time[]+1], zavg, label = "GP")
  plot!(exact,data.z, legend = :topleft, label = "LES", xlabel = "temperature", ylabel = "depth", title = "day " * day_string, xlims = (19,20))
  # display(p1)

  top = vbox(options, kern, kernel_plots)
  bottom = vbox(σ1, γ1, time)
  ui = vbox(top, bottom) # aligns vertically
# end

# webio_serve(page("/",app), port=8000)
# WebIO.webio_serve(page("/", app), port=8000) # serve on a random port

window = Window()
body!(window, ui)

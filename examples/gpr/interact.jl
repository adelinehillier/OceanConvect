# """
# Interactive exploration of the hyperparameter space using Interact and Blink.
# """

using Interact, Blink

include("GP1.jl")

D = 16 # gridpoints
N = 4  # amount of training data

# hyperparameter slider ranges
γs = -3.0:0.1:3.0
σs = 0.0:0.1:2.0

# smooth = toggle("smooth profile?") # smooth the profile
smooth=false
normalize=true

# file to gather data from
filename = togglebuttons(OrderedDict("general_strat_16_profiles" =>"general_strat_16_profiles.jld2",
                                     "general_strat_32_profiles" =>"general_strat_32_profiles.jld2"),
                                     label="LES")
# which variable to explore
V_name = togglebuttons(Dict("T" =>"Temperature [°C]",
                            "wT"=>"Temperature flux [°C⋅m/s]"),
                            label="profile")

γ1 = slider(γs, label="log length scale, log₁₀(γ)") # hyperparameter knob
σ1 = slider(σs, label="log signal variance, log₁₀(σ²)") # hyperparameter knob
time_slider = slider(1:40:(1000), label="time [s]") # time [s]

# distance metric
dist_metric = tabulator(OrderedDict("l²-norm"  =>  "l²-norm:  d(x,x') = || x - x' ||",
                                    "H¹-norm"  =>  "H¹-norm:  d(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||",
                                    "H⁻¹-norm" =>  "H⁻¹-norm: d(x,x') = || diff(x).*diff(z) - diff(x').*diff(z) ||"
                                    ))
# kernel choice
kern = tabulator(OrderedDict("Squared exponential"       => "Squared exponential kernel:           k(x,x') = σ * exp( - ||x-x'||² / 2γ² )",
                             "Matern 1/2"                => "Matérn with ʋ=1/2:                    k(x,x') = σ * exp( - ||x-x'|| / γ )",
                             "Matern 3/2"                => "Matérn with ʋ=3/2:                    k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)",
                             "Matern 5/2"                => "Matérn with ʋ=5/2:                    k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)",
                             "Rational quadratic w/ α=1" => "Rational quadratic kernel:            k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",
                            ))

function get_data(filename::String, V_name)
    if V_name=="Temperature [°C]"; name="T" end
    if V_name=="Temperature flux [°C⋅m/s]"; name="wT" end
    data = construct_profile_data(filename, name, D; N=N, verbose=true)
    return data
end

function get_d(dist_metric)
    if dist_metric==1; return l2_norm end
    if dist_metric==2; return h1_norm end
    if dist_metric==3; return hm1_norm
    else
        throw(error())
    end
end

function get_kernel(k::Int64, γ, σ)
    # convert from log10 scale
    γ = 10^γ
    σ = 10^σ
  if k==1;     return SquaredExponentialI(γ, σ)
  elseif k==2; return Matern12I(γ, σ)
  elseif k==3; return Matern32I(γ, σ)
  elseif k==4; return Matern52I(γ, σ)
  elseif k==5; return RationalQuadraticI(γ, σ, 1.0)
  else; throw(error())
  end
end

function plot_kernel(kernel::Kernel, d, data::ProfileData)
    kmat = [kernel_function(kernel; d=d, z=data.zavg)(i,j) for i in 1:10:data.Nt, j in 1:10:data.Nt]# fill kernel mx with values
    return heatmap(kmat, title = "Covariance Matrix", xaxis=(:false), yaxis=(:flip, :false), clims=(0.0,100), legend=true)
end

#updating variables
#output               function                 args
data            = map(get_data,                filename, V_name)
k               = map(get_kernel,              kern, γ1, σ1)
d               = map(get_d,                   dist_metric)
k_plot          = map(plot_kernel,             k, d, data)
gp              = map(get_gp,                  data, k, d)
gpr_prediction  = map(get_gpr_pred,            gp, data)
profile_plot    = map(plot_profile,            gp, data, V_name, time_slider, gpr_prediction)
log_error_plot  = map(plot_error_histogram,    gp, data, time_slider)
hyp_landscape   = map(error_metric_comparison, kern, data, d, γs)

# layout
top    = vbox(hbox(filename, V_name), hbox(kern, dist_metric), hbox(k_plot, hyp_landscape))
middle = vbox(γ1, σ1, time_slider)
bottom = hbox(profile_plot, log_error_plot) # aligns horizontally
ui     = vbox(top, middle, bottom) # aligns vertically

# Blink GUI
window = Window()
body!(window, ui)

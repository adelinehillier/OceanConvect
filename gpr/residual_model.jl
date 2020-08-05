"""
Residual model to pick up the slack between approximation of -∂z(wT) and the truth.
This example uses the ResidualData struct and GP.

    # temperature eq.
    # ∂t(T) + ∂x(uT) + ∂y(vT) + ∂z(wT) = 𝓀∇²T
    # --> horizontal average -->
    #
    #           diffusive  advective
    # ∂t(T) = - ∂z(wT) +  𝓀∂z(∂z(T))
    #
    # GOAL: express wT in terms of only large-scale terms wT = F(T,h,𝓀,μ)
    #  - 𝓀: diffusivity
    #  - μ: coefficient of viscosity

    # T(n+1) = T(n) + Δt * (- ∂z(wT) +  𝓀∂z(∂z(T)))
    # Gp = -∂z(wT) <- model this
    # y = T(n+1) - T(n) - Δt * ∂z(∂z(T)) <- or model this

    use GPR to capture the difference between the truth
        Δt(-∂z(wT))
    and the approximation
        T(n+1) - T(n) - Δt * ∂z(∂z(T))

    # target
        approx - truth


"""

using Statistics, LinearAlgebra, Plots

const save_figure = false

include("kernels.jl")
include("profile_gpr_sandbox.jl")

D = 16 # gridpoints
log_γs = -3.0:0.1:3.0 # hyperparameter slider range
log_σs = 0.0:0.1:2.0

d = l2_norm

# file to gather data from
# filename = "general_strat_16_profiles.jld2"
filename = "general_strat_32_profiles.jld2"

# which variable to explore
V_name = Dict("T" =>"Temperature [°C]", "wT"=>"Temperature flux [°C⋅m/s]")

smooth = false # smooth the profile
normalize = true # normalize the data (pre- / postprocessing)

# distance metric
dist_metric = Dict(1  =>  "l²-norm:  d(x,x') = || x - x' ||",
                   2  =>  "H¹-norm:  d(x,x') = || diff(x)./diff(z) - diff(x')./diff(z) ||",
                   3 =>  "H⁻¹-norm: d(x,x') = || diff(x).*diff(z) - diff(x').*diff(z) ||"
                  )

kern = Dict("Squared exponential"           => "Squared exponential kernel:           k(x,x') = σ * exp( - ||x-x'||² / 2γ² )",
            "Matern 1/2"                    => "Matérn with ʋ=1/2:                    k(x,x') = σ * exp( - ||x-x'|| / γ )",
            "Matern 3/2"                    => "Matérn with ʋ=3/2:                    k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)",
            "Matern 5/2"                    => "Matérn with ʋ=5/2:                    k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)",
            "Rational quadratic w/ α=1"     => "Rational quadratic kernel:            k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",
            )

# kernel choice
kern = 4

data = construct_profile_data(filename, "T", D; N=4, verbose=true)

function get_kernel(k::Int64, γ, σ)
    # convert from log10 scale
    γ = 10^γ
    σ = 10^σ
  if k==1; return SquaredExponentialKernelI(γ, σ) end
  if k==2; return Matern12I(γ, σ) end
  if k==3; return Matern32I(γ, σ) end
  if k==4; return Matern52I(γ, σ) end
  if k==5; return RationalQuadraticI(γ, σ, 1.0)
  else; throw(error()) end
end


# log_γs = -0.4:0.001:-0.3
println(get_min_gamma(kern, data, normalize, d, log_γs))

# find the minimizing gamma value then animate
# min_gamma, min_error = get_min_gamma(kern, data, normalize, d, log_γs)
# kernel = get_kernel(kern, min_gamma, 0.0)
# 𝒢 = construct_gpr(data.x_train, data.y_train, kernel; distance_fn=d, z=data.zavg, normalize=normalize);
# gpr_prediction = get_gpr_pred(𝒢, data)

# animation_set = 1:30:(data.Nt-2)
# anim = @animate for i in animation_set
#
#     exact = data.v[:,i+1]
#     day_string = string(floor(Int, data.t[i]/86400))
#     p1 = scatter(gpr_prediction[i+1], data.zavg, label = "GP")
#     plot!(exact, data.z, legend = :topleft, label = "LES", xlabel = "$(V_name["T"])", ylabel = "depth", title = "day " * day_string, xlims = (19,20))
#     display(p1)
#
# end
# if save_figure == true
#     gif(anim, pwd() * "gp_emulator.gif", fps = 15)
#     mp4(anim, pwd() * "gp_emulator.mp4", fps = 15)
# end

##

function error_comparison(k::Int64, data::ProfileData, normalize, d, γs)
    # Mean error on greedy check correlated with mean error on true check?

    mlls = zeros(length(γs)) # mean log marginal likelihood
    mes  = zeros(length(γs)) # mean error (greedy check)
    mets  = zeros(length(γs)) # mean error (true check)

    for i in 1:length(γs)

        kernel = get_kernel(k, γs[i], 0.0)
        𝒢 = construct_gpr(data.x_train, data.y_train, kernel; distance_fn=d, z=data.zavg, normalize=normalize);

        # -----compute mll loss----
        mlls[i] = -1*mean_log_marginal_loss(data.y_train, 𝒢, add_constant=false)

        # -----compute mean error for greedy check (same as in plot log error)----
        total_error = 0.0
        # greedy check
        verification_set = data.verification_set
        for j in eachindex(verification_set)
            test_index = verification_set[j]
            y_prediction = prediction([data.x[test_index]], 𝒢)
            error = l2_norm(y_prediction, data.y[test_index])
            total_error += error
        end
        mes[i] = total_error/length(verification_set)

        # -----compute mean error for true check----
        total_error = 0.0
        gpr_prediction = get_gpr_pred(𝒢, data)
        for i in 1:data.Nt-2
            exact    = data.y[i+1]
            predi    = gpr_prediction[i+1]
            total_error += l2_norm(exact, predi) # euclidean distance
        end
        mets[i] = total_error/(data.Nt-2)
    end

    ylims = ( minimum([minimum(mets), minimum(mes)]) , maximum([maximum(mets), maximum(mes)]) )

    mll_plot = plot(γs, mlls, xlabel="log(γ)", title="negative mean log marginal likelihood, P(y|X)", legend=false, yscale=:log10) # 1D plot: mean log marginal loss vs. γ
    vline!([γs[argmin(mlls)]])
    mes_plot  = plot(γs, mes,  xlabel="log(γ)", title="ME on greedy check, min = $(round(minimum(mes);digits=5))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γs[argmin(mes)]])
    met_plot  = plot(γs, mets,  xlabel="log(γ)", title="ME on true check, min = $(round(minimum(mets);digits=5))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γs[argmin(mets)]])

    return plot(mll_plot, mes_plot, met_plot, layout = @layout [a ; b; c])
end


filename = "general_strat_32_profiles.jld2"
D = 16 # gridpoints
log_γs = -3.0:0.01:3.0 # hyperparameter slider range
data = construct_profile_data(filename, "T", D; N=4, verbose=true)

p = error_comparison(1, data, normalize, l2_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes/SE_γ_landscapes_gs16_l2norm.png")

p = error_comparison(2, data, normalize, l2_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes/M12_γ_landscapes_gs16_l2norm.png")

p = error_comparison(3, data, normalize, l2_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes/M32_γ_landscapes_gs16_l2norm.png")

p = error_comparison(4, data, normalize, l2_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes/M52_γ_landscapes_gs16_l2norm.png")

p = error_comparison(5, data, normalize, l2_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes/RQ_α1_γ_landscapes_gs16_l2norm.png")


##

function mes_plot_file_comparison(k::Int64, normalize, d, γs)
    # Mean error on greedy check correlated with mean error on true check?

    results = Dict()
    for file in ["general_strat_8_profiles.jld2","general_strat_16_profiles.jld2","general_strat_32_profiles.jld2"]

        𝒟 = construct_profile_data(file, "T", 16; N=4)
        mets  = zeros(length(γs)) # mean error (true check)

        for i in 1:length(γs)
            kernel = get_kernel(k, γs[i], 0.0)
            𝒢 = construct_gpr(𝒟.x_train, 𝒟.y_train, kernel; distance_fn=d, z=𝒟.zavg, normalize=normalize);
            # -----compute mean error for true check----
            total_error = 0.0
            gpr_prediction = get_gpr_pred(𝒢, 𝒟)
            for i in 1:𝒟.Nt-2
                exact    = 𝒟.y[i+1]
                predi    = gpr_prediction[i+1]
                total_error += l2_norm(exact, predi) # euclidean distance
            end
            mets[i] = total_error/(𝒟.Nt-2)
        end

        results[file]=mets
    end

    ylims = (0.0005,0.05)

    r1 = results["general_strat_8_profiles.jld2"]
    γ=γs[argmin(r1)]
    p1 = plot(γs, r1, xlabel="log(γ)", ylabel="ME, true check", title="general_strat_8_profiles, log(γ)=$(γ), min = $(round(minimum(r1);digits=3))", legend=false, yscale=:log10, ylims=ylims) # 1D plot: mean log marginal loss vs. γ
    vline!([γ])

    r2 = results["general_strat_16_profiles.jld2"]
    γ=γs[argmin(r2)]
    p2  = plot(γs, r2,  xlabel="log(γ)", ylabel="ME, true check", title="general_strat_16_profiles, log(γ)=$(γ), min = $(round(minimum(r2);digits=3))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γ])

    r3 = results["general_strat_32_profiles.jld2"]
    γ=γs[argmin(r3)]
    p3  = plot(γs, r3,  xlabel="log(γ)", ylabel="ME, true check", title="general_strat_32_profiles, log(γ)=$(γ), min = $(round(minimum(r3);digits=3))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γ])

    return plot(p1, p2, p3, layout = @layout [a ; b; c])
end

p = mes_plot_file_comparison(1, normalize, l2_norm, -0.2:0.001:0.2)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_SE_γ_landscapes_l2norm.png")
p = mes_plot_file_comparison(2, normalize, l2_norm, 3.8:0.001:4.3)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_M12_γ_landscapes_l2norm.png")

p = mes_plot_file_comparison(3, normalize, l2_norm, -0.3:0.001:0.3)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_M32_γ_landscapes_l2norm.png")

p = mes_plot_file_comparison(4, normalize, l2_norm, -0.3:0.01:0.3)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_M52_γ_landscapes_l2norm.png")

p = mes_plot_file_comparison(5, normalize, l2_norm, -0.2:0.001:0.2)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_RQ_α1_γ_landscapes_l2norm.png")

## wide

log_γs = -3.0:0.1:3.0
p = mes_plot_file_comparison(1, normalize, l2_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_SE_γ_landscapes_l2norm.png")

p = mes_plot_file_comparison(2, normalize, l2_norm, 2.0:0.1:5.0)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_M12_γ_landscapes_l2norm.png")

p = mes_plot_file_comparison(3, normalize, l2_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_M32_γ_landscapes_l2norm.png")

p = mes_plot_file_comparison(4, normalize, l2_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_M52_γ_landscapes_l2norm.png")

p = mes_plot_file_comparison(5, normalize, l2_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_RQ_α1_γ_landscapes_l2norm.png")

## wide with h1_norm

log_γs = -3.0:0.1:3.0
p = mes_plot_file_comparison(1, normalize, h1_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_SE_γ_landscapes_h1norm.png")

p = mes_plot_file_comparison(2, normalize, h1_norm, 2.0:0.1:5.0)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_M12_γ_landscapes_h1norm.png")

p = mes_plot_file_comparison(3, normalize, h1_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_M32_γ_landscapes_h1norm.png")

p = mes_plot_file_comparison(4, normalize, h1_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_M52_γ_landscapes_h1norm.png")

p = mes_plot_file_comparison(5, normalize, h1_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_RQ_α1_γ_landscapes_h1norm.png")

## wide with hm1_norm

log_γs = -3.0:0.1:3.0
p = mes_plot_file_comparison(1, normalize, hm1_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_SE_γ_landscapes_hm1norm.png")

p = mes_plot_file_comparison(2, normalize, hm1_norm, 2.0:0.1:5.0)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_M12_γ_landscapes_hm1norm.png")

p = mes_plot_file_comparison(3, normalize, hm1_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_M32_γ_landscapes_hm1norm.png")

p = mes_plot_file_comparison(4, normalize, hm1_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_M52_γ_landscapes_hm1norm.png")

p = mes_plot_file_comparison(5, normalize, hm1_norm, log_γs)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims/compare_les_wide_RQ_α1_γ_landscapes_hm1norm.png")

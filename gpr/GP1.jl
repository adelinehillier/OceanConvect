"""
Includes all useful functions for applying GPR to Oceananigans profiles.
Uses ProfileData struct to store data and GP struct for performing GPR on the data.
"""

using JLD2, Statistics, LinearAlgebra, Plots

include("gaussian_process.jl")
include("profile_data.jl")
include("kernels.jl")
include("distance_metrics.jl")
include("scalings.jl")
include("pre_post_processing.jl")
include("../les/get_les_data.jl")

export construct_profile_data
export plot_profile

# data = construct_profile_data(filename, name, D; N=4, verbose=true)

function get_gp(data::ProfileData, kernel::Kernel, d)
    # create instance of GP using data
    𝒢 = construct_gpr(data.x_train, data.y_train, kernel; distance_fn=d, z=data.zavg);
    return 𝒢
end

"""
get_gpr_pred(gp::GP, data::ProfileData; scaled_up=true)
Predict temperature profile from start to finish without the training data.
"""
function get_gpr_pred(gp::GP, data::ProfileData; unscaled=true)
    gpr_prediction = similar(data.vavg[1:data.Nt-1])
    starting = data.x[1] # x is scaled
    gpr_prediction[1] = starting
    for i in 1:(data.Nt-2)
        x = gpr_prediction[i]
        scaled_pred = prediction(x, gp)
        gpr_prediction[i+1] = postprocess_prediction(x, scaled_pred, data.processor)
    end

    if unscaled
        # post-processing - unscale the data for plotting (false for calculating loss)
        gpr_prediction = [unscale(vec, data.scaling) for vec in gpr_prediction]
    end
    return gpr_prediction
end

function plot_profile(gp::GP, data::ProfileData, V_name, time_index, gpr_prediction)
    exact = data.v[:,time_index+1]
    day_string = string(floor(Int, data.t[time_index]/86400))

    if V_name == "Temperature [°C]"; xlims=(19,20) end
    if V_name == "Temperature flux [°C⋅m/s]"; xlims=(-1e-5,4e-5) end
    p = scatter(gpr_prediction[time_index+1], data.zavg, label = "GP", xlims=xlims)
    plot!(exact, data.z, legend = :topleft, label = "LES", xlabel = V_name, ylabel = "depth", title = "day " * day_string)
    return p
end

# kernel options
#  1   =>   "Squared exponential"         => "Squared exponential kernel:        k(x,x') = σ * exp( - ||x-x'||² / 2γ² )",
#  2   =>   "Matern 1/2"                  => "Matérn with ʋ=1/2:                 k(x,x') = σ * exp( - ||x-x'|| / γ )",
#  3   =>   "Matern 3/2"                  => "Matérn with ʋ=3/2:                 k(x,x') = σ * (1+c) * exp(-√(3)*||x-x'||)/γ)",
#  4   =>   "Matern 5/2"                  => "Matérn with ʋ=5/2:                 k(x,x') = σ * ( 1 + √(5)*||x-x'||)/γ + 5*||x-x'||²/(3*γ^2) ) * exp(-√(5)*||x-x'||)/γ)",
#  5   =>   "Rational quadratic w/ α=1"   => "Rational quadratic kernel:         k(x,x') = σ * (1+(x-x')'*(x-x')/(2*α*(γ²))^(-α)",

function get_kernel(k::Int64, γ, σ)
    # convert from log10 scale
    γ = 10^γ
    σ = 10^σ
  if k==1; return SquaredExponentialI(γ, σ) end
  if k==2; return Matern12I(γ, σ) end
  if k==3; return Matern32I(γ, σ) end
  if k==4; return Matern52I(γ, σ) end
  if k==5; return RationalQuadraticI(γ, σ, 1.0)
  else; throw(error()) end
end

#  *--*--*--*--*--*--*--*--*--*--*--*--*--*
#  | Visualize Hyperparameter landscapes  |
#  *--*--*--*--*--*--*--*--*--*--*--*--*--*

function get_me_true_check(𝒢::GP, 𝒟::ProfileData)
    # mean error on true check for a single value of γ
    # computed on the scaled down (range [0,1]) profile values
    total_error = 0.0
    gpr_prediction = get_gpr_pred(𝒢, 𝒟; unscaled=false)
    n = 𝒟.Nt-2
    for i in 1:n
        exact    = 𝒟.y[i+1]
        predi    = gpr_prediction[i+1]
        total_error += l2_norm(exact, predi) # euclidean distance
    end
    return total_error / n
end

function get_me_greedy_check(𝒢::GP, 𝒟::ProfileData)
    # mean error on greedy check for a single value of γ
    # computed on the scaled down (range [0,1]) profile values
    total_error = 0.0
    n = length(𝒟.verification_set)
    # greedy check
    verification_set = 𝒟.verification_set
    for j in 1:n
        test_index = verification_set[j]
        y_prediction = prediction(𝒟.x[test_index], 𝒢)
        error = l2_norm(y_prediction, 𝒟.y[test_index])
        total_error += error
    end
    return total_error / n
end

function error_metric_comparison(k::Int64, 𝒟::ProfileData, d, γs)
    # Mean error on greedy check correlated with mean error on true check?

    mlls = zeros(length(γs)) # mean log marginal likelihood
    mes  = zeros(length(γs)) # mean error (greedy check)
    mets  = zeros(length(γs)) # mean error (true check)

    for i in 1:length(γs)

        kernel = get_kernel(k, γs[i], 0.0)
        𝒢 = construct_gpr(𝒟.x_train, 𝒟.y_train, kernel; distance_fn=d, z=𝒟.zavg);

        # -----compute mll loss----
        mlls[i] = -1*mean_log_marginal_loss(𝒟.y_train, 𝒢, add_constant=false)

        # -----compute mean error for greedy check (same as in plot log error)----
        mes[i] = get_me_greedy_check(𝒢, 𝒟)

        # -----compute mean error for true check----
        mets[i] = get_me_true_check(𝒢, 𝒟)

    end

    ylims = ( minimum([minimum(mets), minimum(mes)]) , maximum([maximum(mets), maximum(mes)]) )

    mll_plot = plot(γs, mlls, xlabel="log(γ)", title="negative mean log marginal likelihood, P(y|X)", legend=false, yscale=:log10) # 1D plot: mean log marginal loss vs. γ
    vline!([γs[argmin(mlls)]])
    mes_plot  = plot(γs, mes,  xlabel="log(γ)", title="ME on greedy check, min = $(round(minimum(mes);digits=7))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γs[argmin(mes)]])
    met_plot  = plot(γs, mets,  xlabel="log(γ)", title="ME on true check, min = $(round(minimum(mets);digits=7))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γs[argmin(mets)]])

    return plot(mll_plot, mes_plot, met_plot, layout = @layout [a ; b; c])
end


function me_file_comparison(k::Int64, d, γs, v_str; N=4)
    # true check mean error for gs8 vs gs16 vs gs32

    results = Dict()
    for file in ["general_strat_8_profiles.jld2","general_strat_16_profiles.jld2","general_strat_32_profiles.jld2"]

        𝒟 = construct_profile_data(file, v_str, 16; N=N)

        mets  = zeros(length(γs))
        for i in 1:length(γs)
            kernel = get_kernel(k, γs[i], 0.0)
            𝒢 = construct_gpr(𝒟.x_train, 𝒟.y_train, kernel; distance_fn=d, z=𝒟.zavg);
            mets[i] = get_me_true_check(𝒢, 𝒟)
        end

        results[file]=mets
    end

    # ylims = (0.0005,0.05)

    r1 = results["general_strat_8_profiles.jld2"]
    r2 = results["general_strat_16_profiles.jld2"]
    r3 = results["general_strat_32_profiles.jld2"]
    ylims = (minimum(minimum([r1,r2,r3])),maximum(maximum([r1,r2,r3])))

    γ=γs[argmin(r1)]
    p1 = plot(γs, r1, xlabel="log(γ)", ylabel="ME, true check", title="general_strat_8_profiles, log(γ)=$(γ), min = $(round(minimum(r1);digits=5))", legend=false, yscale=:log10, ylims=ylims) # 1D plot: mean log marginal loss vs. γ
    vline!([γ])

    γ=γs[argmin(r2)]
    p2  = plot(γs, r2,  xlabel="log(γ)", ylabel="ME, true check", title="general_strat_16_profiles, log(γ)=$(γ), min = $(round(minimum(r2);digits=5))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γ])

    γ=γs[argmin(r3)]
    p3  = plot(γs, r3,  xlabel="log(γ)", ylabel="ME, true check", title="general_strat_32_profiles, log(γ)=$(γ), min = $(round(minimum(r3);digits=5))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γ])

    return plot(p1, p2, p3, layout = @layout [a ; b; c])
end


function me_file_comparison2(k::Int64, d, γs, v_str; N=4)
    # true check mean error for gs8 vs gs16 vs gs32

    results = Dict()
    for file in ["general_strat_8_profiles.jld2","general_strat_16_profiles.jld2","general_strat_24_profiles.jld2","general_strat_32_profiles.jld2"]

        𝒟 = construct_profile_data(file, v_str, 16; N=N)

        mets  = zeros(length(γs))
        for i in 1:length(γs)
            kernel = get_kernel(k, γs[i], 0.0)
            𝒢 = construct_gpr(𝒟.x_train, 𝒟.y_train, kernel; distance_fn=d, z=𝒟.zavg);
            mets[i] = get_me_true_check(𝒢, 𝒟)
        end

        results[file]=mets
    end

    # ylims = (0.0005,0.05)

    r1 = results["general_strat_8_profiles.jld2"]
    r2 = results["general_strat_16_profiles.jld2"]
    r3 = results["general_strat_24_profiles.jld2"]
    r4 = results["general_strat_32_profiles.jld2"]
    ylims = (minimum(minimum([r1,r2,r3])),maximum(maximum([r1,r2,r3])))

    γ=γs[argmin(r1)]
    p1 = plot(γs, r1, title="gs8, log(γ)=$(γ), min = $(round(minimum(r1);digits=6))", legend=false, yscale=:log10, ylims=ylims) # 1D plot: mean log marginal loss vs. γ
    vline!([γ])

    γ=γs[argmin(r2)]
    p2  = plot(γs, r2, title="gs16, log(γ)=$(γ), min = $(round(minimum(r2);digits=6))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γ])

    γ=γs[argmin(r3)]
    p3  = plot(γs, r3, ylabel="Mean Error, True Check", title="gs24, log(γ)=$(γ), min = $(round(minimum(r3);digits=6))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γ])

    γ=γs[argmin(r4)]
    p4  = plot(γs, r4, xlabel="log(γ)",  title="gs32, log(γ)=$(γ), min = $(round(minimum(r4);digits=6))", legend=false, yscale=:log10, ylims=ylims)  # 1D plot: mean error vs. γ
    vline!([γ])

    return plot(p1, p2, p3, p4, layout = @layout [a ; b; c; d])
end


function plot_error_histogram(gp::GP, data::ProfileData, time_index)
    #compute error for true check
    gpr_prediction = get_gpr_pred(gp, data)
    gpr_error = zeros(data.Nt-2)
    for i in 1:data.Nt-2
        exact    = data.y[i+1]
        predi    = gpr_prediction[i+1]
        gpr_error[i] = l2_norm(exact, predi) # euclidean distance
    end
    mean_error = sum(gpr_error)/(data.Nt-2)

    error_plot_log = histogram(log.(gpr_error), title = "log(error) at each timestep of the full evolution", xlabel="log(Error)", ylabel="Frequency",ylims=(0,250), label="frequency")
    vline!([log(mean_error)], line = (4, :dash, 0.8), label="mean error")
    vline!([log(gpr_error[time_index])], line = (1, :solid, 0.6), label="error at t=$(time_index)")
end

function get_min_gamma(k::Int64, data::ProfileData, d, γs)
    # returns the gamma value that minimizes the mean error on the true check
    # only tests the gamma values listed in γs parameter

    mets  = zeros(length(γs)) # mean error for each gamma (true check)
    for i in eachindex(γs)

        kernel = get_kernel(k, γs[i], 0.0)
        𝒢 = construct_gpr(data.x_train, data.y_train, kernel; distance_fn=d, z=data.zavg);

        # -----compute mean error for true check----
        total_error = 0.0
        gpr_prediction = get_gpr_pred(𝒢, data)
        for q in 1:data.Nt-2
            exact        = data.y[q+1]
            predi        = gpr_prediction[q+1]
            total_error += l2_norm(exact, predi) # euclidean distance
        end
        mets[i] = total_error/(data.Nt-2)
    end

    i = argmin(mets)
    min_gamma = γs[i]
    min_error = mets[i]

    return (min_gamma, min_error)
end

function get_min_gamma_alpha(k::Int64, data::ProfileData, d, γs)
    # returns the gamma value that minimizes the mean error on the true check
    # only tests the gamma values listed in γs parameter

    mets  = zeros(length(γs*αs)) # mean error for each gamma (true check)
    for i in eachindex(γs)
        for j in eachindex(αs)

        kernel = RationalQuadraticI(γ[i], 0.0, αs[j])
        𝒢 = construct_gpr(data.x_train, data.y_train, kernel; distance_fn=d, z=data.zavg);

        # -----compute mean error for true check----
        total_error = 0.0
        gpr_prediction = get_gpr_pred(𝒢, data)
        for index in 1:data.Nt-2
            exact        = data.y[index+1]
            predi        = gpr_prediction[index+1]
            total_error += l2_norm(exact, predi) # euclidean distance
        end
        mets[i] = total_error/(data.Nt-2)
    end
    end

    i = argmin(mets)
    min_gamma = γs[i]
    min_error = mets[i]

    return (min_gamma, min_error)
end

# function plot_hyp_landscp(k::Int64, data::ProfileData, normalize, d, γs)
#
#     mlls = zeros(length(γs)) # mean log marginal likelihood
#     mes  = zeros(length(γs)) # mean error (greedy check)
#     mets  = zeros(length(γs)) # mean error (true check)
#
#     for i in 1:length(γs)
#
#         kernel = get_kernel(k, γs[i], 0.0)
#         𝒢 = construct_gpr(data.x_train, data.y_train, kernel; distance_fn=d, z=data.zavg);
#
#         # -----compute mll loss----
#         mlls[i] = -1*mean_log_marginal_loss(data.y_train, 𝒢, add_constant=false)
#
#         # -----compute mean error for greedy check (same as in plot log error)----
#         total_error = 0.0
#         # greedy check
#         verification_set = data.verification_set
#         for j in eachindex(verification_set)
#             test_index = verification_set[j]
#             y_prediction = prediction([data.x[test_index]], 𝒢)
#             error = l2_norm(y_prediction, data.y[test_index])
#             total_error += error
#         end
#         mes[i] = total_error/length(verification_set)
#
#         # -----compute mean error for true check----
#         total_error = 0.0
#         gpr_prediction = get_gpr_pred(𝒢, data)
#         for i in 1:data.Nt-2
#             exact    = data.y[i+1]
#             predi    = gpr_prediction[i+1]
#             total_error += l2_norm(exact, predi) # euclidean distance
#         end
#         mets[i] = total_error/(data.Nt-2)
#
#     end
#
#     mll_plot = plot(γs, mlls, xlabel="log(γ)", title="negative mean log marginal likelihood, P(y|X)", legend=false, yscale=:log10) # 1D plot: mean log marginal loss vs. γ
#     # me_plot  = plot(γs, mets,  xlabel="log(γ)", title="mean error on full evolution ('true check'), min = $(minimum(mets))", legend=false, yscale=:log10)  # 1D plot: mean error vs. γ
#     me_plot  = plot(γs, mets,  xlabel="log(γ)", title="min = $(minimum(mets))", legend=false, yscale=:log10)  # 1D plot: mean error vs. γ
#
#     return plot(mll_plot, me_plot, layout = @layout [a ; b])
# end

#  *--*--*--*--*
#  | ANIMATE   |
#  *--*--*--*--*

function animate_profile(filename, k::Int64, γ, d, D, V_str; N=4)

    V_name = Dict("T" =>"Temperature [°C]", "wT"=>"Temperature flux [°C⋅m/s]")
    x_lims = Dict("T" =>(18,20), "wT"=>(-1e-5,4e-5))

    xlims = x_lims[V_str]

    data = construct_profile_data(filename, V_str, D; N=N)
    # find the minimizing gamma value then animate
    # min_gamma, min_error = get_min_gamma(2, data, normalize, l2_norm, log_γs)
    kernel = get_kernel(k, γ, 0.0)
    𝒢 = construct_gpr(data.x_train, data.y_train, kernel; distance_fn=d, z=data.zavg);
    gpr_prediction = get_gpr_pred(𝒢, data)

    animation_set = 1:30:(data.Nt-2)
    anim = @animate for i in animation_set
        exact = data.v[:,i]
        day_string = string(floor(Int, data.t[i]/86400))
        p1 = scatter(gpr_prediction[i], data.zavg, label = "GP")
        plot!(exact, data.z, legend = :topleft, label = "LES", xlabel = "$(V_name[V_str])", ylabel = "depth", title = "day " * day_string, xlims=xlims)
    end

    return anim
end

"""
This example does uses the ProfileData struct and GP.
"""

include("GP1.jl")

log_γs = -3.0:0.1:3.0
log_σs = 0.0:0.1:2.0

V_name = Dict("T" =>"Temperature [°C]", "wT"=>"Temperature flux [°C⋅m/s]")

v_str = "T"
# v_str = "wT"

N = 4
# N = 2

##
#  *--*--*--*--*--*--*--*--*--*
#  | Error metric comparison  |
#  *--*--*--*--*--*--*--*--*--*

filename = "general_strat_16_profiles.jld2"
data = construct_profile_data(filename, v_str, 16; N=N)
log_γs = -3.0:0.1:3.0 # hyperparameter slider range

for k in 1:5
    #with normalization
    p = error_metric_comparison(k, data, l2_norm, log_γs, true)
    savefig(pwd() * "/hyperparameter_landscapes/with_normalization_$(v_str)/kernel$(k)_γ_gs16_l2norm.png")
    #no normalization
    p = error_metric_comparison(k, data, l2_norm, log_γs, false)
    savefig(pwd() * "/hyperparameter_landscapes/no_normalization_$(v_str)/kernel$(k)_γ_gs16_l2norm.png")
end

##
#  *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
#  | ME on true check for gs8 vs gs16 vs gs32: fixed range of γs  |
#  *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*

# ------- L2 norm -------
for k in 1:5

    if k==2; log_γs = 0.0:0.1:6.0
    else; log_γs = -3.0:0.1:3.0
    end

    #with normalization
    p = me_file_comparison(k, l2_norm, log_γs, v_str; normalize=true)
    savefig(pwd() * "/hyperparameter_landscapes/with_normalization_$(v_str)/compare_sims_kernel$(k)_γ_l2norm.png")
    #no normalization
    p = me_file_comparison(k, l2_norm, log_γs, v_str; normalize=false)
    savefig(pwd() * "/hyperparameter_landscapes/no_normalization_$(v_str)/compare_sims_kernel$(k)_γ_l2norm.png")
end
# ------- H1 norm -------
for k in 1:5

    if k==2; log_γs = 0.0:0.1:6.0
    else; log_γs = -3.0:0.1:3.0
    end

    #with normalization
    p = me_file_comparison(k, h1_norm, log_γs, v_str; normalize=true)
    savefig(pwd() * "/hyperparameter_landscapes/with_normalization_$(v_str)/compare_sims_kernel$(k)_γ_h1norm.png")
    #no normalization
    p = me_file_comparison(k, h1_norm, log_γs, v_str; normalize=false)
    savefig(pwd() * "/hyperparameter_landscapes/no_normalization_$(v_str)/compare_sims_kernel$(k)_γ_h1norm.png")
end

# ------- H^-1 norm -------
for k in 1:5

    if k==2; log_γs = 0.0:0.1:6.0
    else; log_γs = -3.0:0.1:3.0
    end

    #with normalization
    p = me_file_comparison(k, hm1_norm, log_γs, v_str; normalize=true)
    savefig(pwd() * "/hyperparameter_landscapes/with_normalization_$(v_str)/compare_sims_kernel$(k)_γ_hm1norm.png")
    #no normalization
    p = me_file_comparison(k, hm1_norm, log_γs, v_str; normalize=false)
    savefig(pwd() * "/hyperparameter_landscapes/no_normalization_$(v_str)/compare_sims_kernel$(k)_γ_hm1norm.png")
end

##
#  *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
#  | ME on true check for gs8 vs gs16 vs gs32: neighborhood of the min  |
#  *--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*
# custom γ ranges
# l2_norm

p = me_file_comparison(1, l2_norm, 3.8:0.001:4.3, v_str; normalize=true)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims_with_normalization_$(v_str)/compare_les_SE_γ_landscapes_l2norm.png")

p = me_file_comparison(2, l2_norm, 3.8:0.001:4.3, v_str; normalize=true)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims_with_normalization_$(v_str)/compare_les_M12_γ_landscapes_l2norm.png")

p = me_file_comparison(3, l2_norm, -0.3:0.001:0.3, v_str; normalize=true)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims_with_normalization_$(v_str)/compare_les_M32_γ_landscapes_l2norm.png")

p = me_file_comparison(4, l2_norm, -0.3:0.01:0.3, v_str; normalize=true)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims_with_normalization_$(v_str)/compare_les_M52_γ_landscapes_l2norm.png")

p = me_file_comparison(5, l2_norm, -0.2:0.001:0.2, v_str; normalize=true)
savefig(pwd() * "/plots/hyperparameter_landscapes_constlims_with_normalization_$(v_str)/compare_les_RQ_α1_γ_landscapes_l2norm.png")

##
#  *--*--*--*--*--*--*
#  | Animations      |
#  *--*--*--*--*--*--*

function animate_profile_trained_on_gs16(filename, k::Int64, γ, d, D)

    mydata = construct_profile_data(filename, v_str, D; N=N)

    # find the minimizing gamma value then animate
    # min_gamma, min_error = get_min_gamma(2, data, normalize, l2_norm, log_γs)
    kernel = get_kernel(k, γ, 0.0)

    data16 = construct_profile_data("general_strat_16_profiles.jld2", v_str, D; N=N, verbose=true)
    𝒢 = construct_gpr(data16.x_train, data16.y_train, kernel; distance_fn=d, z=data16.zavg, normalize=normalize);

    # 𝒢 = construct_gpr(data.x_train, data.y_train, kernel; distance_fn=d, z=data.zavg, normalize=normalize);
    gpr_prediction = get_gpr_pred(𝒢, mydata)

    # println(mydata.x[1])

    println(prediction([collect(1:16)], 𝒢))


    animation_set = 1:10:(mydata.Nt-2)
    anim = @animate for i in animation_set
        exact = mydata.v[:,i]
        day_string = string(floor(Int, mydata.t[i]/86400))
        p1 = scatter(gpr_prediction[i], mydata.zavg, label = "GP")
        # xlims=(minimum(data.v[:,1]),maximum(data.v[:,1]))
        xlims=(18,20)

        if i<data16.Nt
            exact16 = data16.v[:,i]
        else
            exact16 = data16.v[:,data16.Nt]
        end

        plot!(exact, mydata.z, legend = :topleft, label = "LES", xlabel = "$(V_name["T"])", ylabel = "depth", title = "i = $(i)", xlims=xlims)
        plot!(exact16, data16.z, legend = :topleft, label = "LES gs 16", xlabel = "$(V_name["T"])", ylabel = "depth", xlims=xlims)

    end

    return anim
end

filename="general_strat_8_profiles"
anim = animate_profile("$(filename).jld2", 2, 4.1, l2_norm, 16)
gif(anim, pwd() * "/../les/data_sandreza/$(filename)/gp_γ4.1_M12_l2norm_trainedOnGs16.gif", fps = 20)

## compare distance metrics

# function mes_plot_file_comparison(k::Int64, filename, normalize, d, γs)
#     # Mean error on greedy check correlated with mean error on true check?
#
#     results = Dict()
#     for dm in [l2_norm, h1_norm, hm1_norm]
#
#         𝒟 = construct_profile_data(file, "T", 16; N=N)
#         mets  = zeros(length(γs)) # mean error (true check)
#
#         for i in 1:length(γs)
#             kernel = get_kernel(k, γs[i], 0.0)
#             𝒢 = construct_gpr(𝒟.x_train, 𝒟.y_train, kernel; distance_fn=dm, z=𝒟.zavg, normalize=normalize);
#             # -----compute mean error for true check----
#             total_error = 0.0
#             gpr_prediction = get_gpr_pred(𝒢, 𝒟)
#             for i in 1:𝒟.Nt-2
#                 exact    = 𝒟.y[i+1]
#                 predi    = gpr_prediction[i+1]
#                 total_error += l2_norm(exact, predi) # euclidean distance
#             end
#             mets[i] = total_error/(𝒟.Nt-2)
#         end
#
#         results[dm]=mets
#     end
#
#     r1 = results[l2_norm]
#     γ=γs[argmin(r1)]
#     p1 = plot(γs, r1, xlabel="log(γ)", ylabel="ME, true check", title="l² norm, log(γ)=$(γ), min = $(round(minimum(r1);digits=5))", legend=false, yscale=:log10) # 1D plot: mean log marginal loss vs. γ
#     vline!([γ])
#
#     r2 = results[h1_norm]
#     γ=γs[argmin(r2)]
#     p2  = plot(γs, r2,  xlabel="log(γ)", ylabel="ME, true check", title="H¹ norm, log(γ)=$(γ), min = $(round(minimum(r2);digits=5))", legend=false, yscale=:log10)  # 1D plot: mean error vs. γ
#     vline!([γ])
#
#     r3 = results[hm1_norm]
#     γ=γs[argmin(r3)]
#     p3  = plot(γs, r3,  xlabel="log(γ)", ylabel="ME, true check", title="H⁻¹ norm, log(γ)=$(γ), min = $(round(minimum(r3);digits=5))", legend=false, yscale=:log10)  # 1D plot: mean error vs. γ
#     vline!([γ])
#
#     return plot(p1, p2, p3, layout = @layout [a ; b; c])
# end



## automate finding gamma that minimizes the mean error

# filename = "general_strat_16_profiles.jld2"
filename = "general_strat_32_profiles.jld2"

# kernel choice
kern = 4

data = construct_profile_data(filename, "T", D; N=N, verbose=true)

kern =
d =

# find the minimizing gamma value then animate
min_gamma, min_error = get_min_gamma(kern, data, normalize, d, -0.3:0.1:0.3)
kernel = get_kernel(kern, min_gamma, 0.0)
𝒢 = construct_gpr(data.x_train, data.y_train, kernel; distance_fn=d, z=data.zavg, normalize=normalize);
gpr_prediction = get_gpr_pred(𝒢, data)

animation_set = 1:30:(data.Nt-2)
anim = @animate for i in animation_set

    exact = data.v[:,i+1]
    day_string = string(floor(Int, data.t[i]/86400))
    p1 = scatter(gpr_prediction[i+1], data.zavg, label = "GP")
    plot!(exact, data.z, legend = :topleft, label = "LES", xlabel = "$(V_name[v_str])", ylabel = "depth", title="gs32, k=$(kern), $(d), log(γ)=$(min_gamma), error=$(min_error) day " * day_string, xlims = (-1e-5,4e-5))
    display(p1)

end
save_figure=true
if save_figure == true
    gif(anim, pwd() * "/../les/data_sandreza/$(filename)/gp_γ4.1_M12_l2norm_trainedOnGs16.gif", fps = 15)
end

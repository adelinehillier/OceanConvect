using GaussianProcess

# Construct a ProfileData object, 𝒟, consisting of the data to train on,
# and a GP object, 𝒢, with which to perform the regression.

# data
filename = "general_strat_4_profiles.jld2" # "free_convection_day0.25_profiles.nc"
v        = "T"      # v: "T" for temperature profile; "wT" for temperature flux profile
D        = 16       # collapse profile data down to 16 gridpoints
N        = 4        # collect every 4 timesteps' data for training

# kernel
d        = l2_norm  # distance metric to use in the kernel
log_γ    = 4.1      # log(length-scale parameter)
k        = 2        # kernel function ID
kernel   = get_kernel(k, log_γ, 0.0)

𝒟 = construct_profile_data(filename, v, D; N=N)
𝒢 = get_gp(𝒟, kernel, d)

# Find the log(length-scale parameter) value in the range -3:0.1:3 that minimizes the mean error on the true check
min_log_param, min_error = get_min_gamma(k, 𝒟, d, -3:0.1:3)

# Animate the mean GP prediction.
animate_profile(𝒢, 𝒟, v)





# α_proxy(x) = x[end] - x[end-1]

# 𝒟4 = construct_profile_data("general_strat_4_profiles.jld2", v_str, 16; N=N)
# am4 = α_proxy(𝒟4.x_train[50])
#
# 𝒟8 = construct_profile_data("general_strat_8_profiles.jld2", v_str, 16; N=N)
# am8 = α_proxy(𝒟8.x_train[1])
# am8 = α_proxy(𝒟8.x_train[50])
# am8 = α_proxy(𝒟8.x_train[133])
#
# 𝒟16 = construct_profile_data("general_strat_16_profiles.jld2", v_str, 16; N=N)
# am16 = α_proxy(𝒟16.x_train[1])
#
# 𝒟24 = construct_profile_data("general_strat_24_profiles.jld2", v_str, 16; N=N)
# am24 = α_proxy(𝒟24.x_train[50])
#
# 𝒟32 = construct_profile_data("general_strat_32_profiles.jld2", v_str, 16; N=N)
# am32 = α_proxy(𝒟32.x_train[50])
#
# p = plot(𝒟4.x_train[50], 𝒟4.zavg, legend=false)
# plot!(𝒟8.x_train[50], 𝒟8.zavg)
# plot!(𝒟16.x_train[50], 𝒟16.zavg)
# plot!(𝒟24.x_train[50], 𝒟24.zavg)
# plot!(𝒟32.x_train[50], 𝒟32.zavg)
#
# 𝒟4.x_train[50]


# gpr_prediction = get_gpr_pred(𝒢, 𝒟)


# animation_set = 1:10:(𝒟.Nt-2)
# anim = @animate for i in animation_set
#     exact = 𝒟.v[:,i]
#     day_string = string(floor(Int, 𝒟.t[i]/86400))
#     p1 = scatter(gpr_prediction[i], 𝒟.zavg, label = "GP")
#     # xlims=(minimum(data.v[:,1]),maximum(data.v[:,1]))
#     xlims=(18,20)
#
#     if i<𝒟.Nt
#         exact16 = 𝒟.v[:,i]
#     else
#         exact16 = 𝒟.v[:,data16.Nt]
#     end
#
#     plot!(exact, 𝒟.z, legend = :topleft, label = "LES", xlabel = "$(V_name["T"])", ylabel = "depth", title = "i = $(i)", xlims=xlims)
#     # plot!(exact16, data16.z, legend = :topleft, label = "LES gs 16", xlabel = "$(V_name["T"])", ylabel = "depth", xlims=xlims)
#
# end
#
# gif(anim, pwd() * "ignore.gif", fps = 20)

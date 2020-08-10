include("GP1.jl")

V_name = Dict("T" =>"Temperature [°C]", "wT"=>"Temperature flux [°C⋅m/s]")

v_str = "T"
# v_str = "wT"

N = 4
d = l2_norm
γ = 4.1
k = 2

filename = "general_strat_4_profiles.jld2"
filename = "free_convection_day0.25_profiles.nc"

# find the minimizing gamma value then animate
# min_gamma, min_error = get_min_gamma(2, data, normalize, l2_norm, log_γs)
kernel = get_kernel(k, γ, 0.0)

𝒟 = construct_profile_data(filename, v_str, 16; N=N)
𝒢 = construct_gpr(𝒟.x_train, 𝒟.y_train, kernel; distance_fn=d, z=𝒟.zavg);



α_proxy(x) = (x[3] - x[1])*1000
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

long_name = Dict("T" =>"Temperature [°C]", "wT"=>"Temperature flux [°C⋅m/s]")
x_lims = Dict("T" =>(18,20), "wT"=>(-1e-5,4e-5))

"""
plot_profile(gp::GP, data::ProfileData, V_name, time_index, gpr_prediction)
----- Description
Plots the simulation temperature profile at a single given index in the data timeseries along with the
corresponding GP prediction. i.e. produces a snapshot of the profile evolution which can be used to
create an animation.
----- Arguments
- '𝒢' (GP). The GP object
- '𝒟' (ProfileData). The ProfileData object whose starting profile will be evolved forward using 𝒢.
- 'V_name' (ProfileData). The ProfileData object whose starting profile will be evolved forward using 𝒢.
- 'time_index' (Int). The time index
- 'gpr_prediction' (Array). Output of get_gpr_pred (which should only be computed once) on 𝒢 and 𝒟.
"""
function plot_profile(𝒢::GP, 𝒟::ProfileData, time_index, gpr_prediction)

    exact = 𝒟.v[:,time_index+1]
    day_string = string(floor(Int, 𝒟.t[time_index]/86400))

    v_str = 𝒟.problem.variable # "T" or "wT"
    xlims = x_lims[v_str]

    p = scatter(gpr_prediction[time_index+1], 𝒟.zavg, label = "GP", xlims=xlims)
    plot!(exact, 𝒟.z, legend = :topleft, label = "LES", xlabel = "$(long_name[v_str])", ylabel = "Depth [m]", title = "day " * day_string, xlims=xlims)
    return p
end

function animate_profile(𝒢, 𝒟)

    v_str = 𝒟.problem.variable # "T" or "wT"
    xlims = x_lims[v_str]

    predi = get_gpr_pred(𝒢, 𝒟)

    animation_set = 1:30:(𝒟.Nt-2)
    anim = @animate for i in animation_set
        exact = 𝒟.v[:,i]
        day_string = string(floor(Int, 𝒟.t[i]/86400))
        scatter(predi[i], 𝒟.zavg, label = "GP")
        plot!(exact, 𝒟.z, legend = :topleft, label = "LES", xlabel = "$(long_name[v_str])", ylabel = "Depth [m]", title = "day " * day_string, xlims=xlims)
    end

    return anim
end

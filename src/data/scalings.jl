# Normalizing the data before pre-processing and after post-processing.

abstract type Scaling end

struct Tscaling <: Scaling
    ΔT::Float64 #range of Temp values in initial profile
    T_max::Float64
end

#scale: normalize the data so that it ranges from [-1,0]
scale(x, scaling::Tscaling) = (x .- scaling.T_max) ./ scaling.ΔT
#unscale: undo normalization
unscale(x, scaling::Tscaling) = x .* scaling.ΔT .+ scaling.T_max

struct wTscaling <: Scaling
    nc::Float64 #normalization constant
end

#scale: divide all wT values by the maximum value across all timesteps and gridpoints
scale(x, scaling::wTscaling) = x ./ scaling.nc
#scale: undo scale
unscale(x, scaling::wTscaling) = x .* scaling.nc

function get_scaling(V_name, vavg)
    if V_name=="T"; return Tscaling(vavg[1][end]-vavg[1][1], maximum(vavg[1])) # Tscaling(ΔT, T_max)
    elseif V_name=="wT"; return wTscaling( maximum(maximum, vavg) ) # wTscaling(nc)
    else; throw(error()) end
end

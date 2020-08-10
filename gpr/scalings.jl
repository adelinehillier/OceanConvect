abstract type Scaling end

struct Tscaling <: Scaling
    ΔT::Float64 #range of Temp values in initial profile
end

#forward: forward mapping from T to T'∈[0,1] (pre-processing).
forward(x, scaling::Tscaling) = (x .- 20.0) ./ scaling.ΔT
#backward: backward mapping from T' to T (post-processing).
backward(x, scaling::Tscaling) = x .* scaling.ΔT .+ 20.0

struct wTscaling <: Scaling
    nc::Float64 #normalization constant
end

forward(x, scaling::wTscaling) = x ./ scaling.nc
backward(x, scaling::wTscaling) = x .* scaling.nc


function get_scaling(V_name, vavg)
    if V_name=="T"; return Tscaling(vavg[1][end]-vavg[1][1]) # Tscaling(ΔT)
    elseif V_name=="wT"; return wTscaling( maximum(maximum, vavg) ) # wTscaling(nc)
    else; throw(error()) end
end

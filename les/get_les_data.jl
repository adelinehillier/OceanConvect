
function get_les_data(filename::String)

    if filename[end-3:end]==".nc"
        include("../les/convert_netcdf_to_data.jl")
        filename="~/Desktop/OceanConvect/les/data/$(filename[1:end-3])/$(filename)"
    else
        include("../les/convert_jld2_to_data.jl")
        filename=homedir()*"/Desktop/OceanConvect/les/data_sandreza/$(filename[1:end-5])/$(filename)"
    end

    return OceananigansData(filename)
end

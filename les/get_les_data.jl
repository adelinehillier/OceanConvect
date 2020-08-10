
function get_les_data(filename::String)

    include("../les/output_to_data.jl")

    if filename[end-2:end]==".nc"
        println("Collecting NetCDF data.")
        # include("../les/convert_netcdf_to_data.jl")
        filename=homedir()*"/Desktop/OceanConvect/les/data/$(filename[1:end-3])/$(filename)"
        return ReadNetCDF_OceananigansData(filename)
    else
        println("Collecting JLD2 data.")
        # include("../les/convert_jld2_to_data.jl")
        filename=homedir()*"/Desktop/OceanConvect/les/data_sandreza/$(filename[1:end-5])/$(filename)"
        return ReadJLD2_OceananigansData(filename)
    end

    # return  Base.invokelatest(OceananigansData(filename))
end

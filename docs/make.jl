using Documenter, OceanConvect

makedocs(sitename="OceanConvect")

example_pages = [
    "One-dimensional diffusion"        => "generated/one_dimensional_diffusion.md",
    "Two-dimensional turbulence"       => "generated/two_dimensional_turbulence.md",
    "Ocean wind mixing and convection" => "generated/ocean_wind_mixing_and_convection.md",
    "Ocean convection with plankton"   => "generated/ocean_convection_with_plankton.md",
    "Internal wave"                    => "generated/internal_wave.md",
    "Langmuir turbulence"              => "generated/langmuir_turbulence.md"
]

format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"

makedocs(
    ...,
    pages = [
        "page.md",
        "Page title" => "page2.md",
        "Subsection" => [
            ...
        ]
    ]
)

deploydocs(
    repo = "github.com/USER_NAME/PACKAGE_NAME.jl.git",
)




using Documenter
using OceanConvect


dg_methods = Any[
    "Home" => "dg_methods.md",
    "Single Element" => "dg_single_element.md",
    "Boundary Conditions" => "boundary_conditions.md",
    "Variational Crimes" => "inexact_quadrature.md"
]

physics = Any[
    "Home" => "physics.md",
    "Convective Adjustment" => "convective_adjustment.md",
]

makedocs(
    sitename = "DG_Playground",
    format = Documenter.HTML(collapselevel = 1),
    pages = [
    "Home" => "index.md",
    "Examples" => examples,
    "GPR" => gpr,
    "Function Index" => "function_index.md",
    ],
    modules = [DG_Playground]
)

deploydocs(repo = "https://github.com/adelinehillier/OceanConvect.git")

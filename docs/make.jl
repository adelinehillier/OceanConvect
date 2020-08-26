using Documenter
using OceanConvect

format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        collapselevel = 1
)

example_pages = [
    "Basic Gaussian Process" => "demos/gpr/demo.md"
]

pages = Any[
"Home" => "index.md",
"Examples" => example_pages,
"Function Index" => "function_index.md",
]

makedocs(
    sitename = "OceanConvect.jl",
    format   = format,
    pages    = pages,
    modules  = [OceanConvect],
    clean    = true,
    doctest  = false
)

deploydocs(
    repo = "https://github.com/adelinehillier/OceanConvect.git"
)

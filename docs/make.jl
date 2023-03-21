using Documenter, GraphNets
makedocs(
    sitename="GraphNets.jl",
    modules=[GraphNets],
)
deploydocs(
    repo = "github.com/JuliaMLTools/GraphNets.jl.git",
)
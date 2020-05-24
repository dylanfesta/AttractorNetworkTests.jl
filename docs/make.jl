using Documenter, AttractorNetworksTests

makedocs(;
    modules=[AttractorNetworksTests],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/dylanfesta/AttractorNetworksTests.jl/blob/{commit}{path}#L{line}",
    sitename="AttractorNetworksTests.jl",
    authors="Dylan Festa",
    assets=String[],
)

deploydocs(;
    repo="github.com/dylanfesta/AttractorNetworksTests.jl",
)

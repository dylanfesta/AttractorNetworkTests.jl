
using Pkg
Pkg.activate(joinpath(@__DIR__(),".."))
using Plots ; theme(:dark)
using AttractorNetworksTests ; const A=AttractorNetworksTests
using LinearAlgebra, Statistics, StatsBase
using Random ; Random.seed!(0)
using Test


##
ne=7
ni=22
n=ne+ni
# stable dynamics, diagonal weights
wdiag = 0.5*rand(n)
M = diagm(0=>wdiag)
currs = 5.0randn(n)
ntw = A.RecurrentNetwork(ne,ni ; gfun=A.B.GFId(), taus=1 .+ rand(n),
    external_current=currs)
copyto!(ntw.weights,M)
attr_an = @. -currs/(wdiag-1)
u0=attr_an

u_error = randn(n)
u_attr = u_error .+ u0
distfun = A.NormError()
expected_err1 = distfun(u_attr,u0)
expected_err2 = distfun(u_attr,u0,ntw) # because g(u)=u here
expected_err3 = norm(u_attr .- u0)^2/n
@test all( isapprox.( [expected_err2,expected_err3] .- expected_err1,0.0, atol=1E-4))
function colmatrix(v::AbstractVector)
    n=length(v)
    T=eltype(v)
    ret = Matrix{T}(undef,n,1)
    ret[:,1] .= v
    return ret
end
_,expected_err4 = A.effective_attractors(colmatrix(u_attr),ntw ; attr_err=distfun )
@test isapprox(expected_err4[1],expected_err1 ; rtol=0.1)



function effective_attractors(u_attr::Matrix,ntw::RecurrentNetwork;
           t_end=80. , veltol::Float64=1E-4,
           attr_err::AttractorError=NormError())

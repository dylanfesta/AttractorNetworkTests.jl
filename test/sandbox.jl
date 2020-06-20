using Plots ; theme(:dark)
using Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using LinearAlgebra
using AttractorNetworksTests ; const A = AttractorNetworksTests
using MultivariateStats
using Statistics

myheatmap(A) = let n=size(A,1);
  heatmap(A;ratio=1,leg=false,xlims=(0.5 .+ [0.,n]),axis=nothing,seriescolor=:thermal) end

function klinrun(k::Integer,x0c::Real,M,t_end)
  xs_all = Matrix{Float64}[]
  t_all = Vector{Float64}[]
  for i in 1:k
    x0= x0c .*randn(size(M,1))
    (t,xs)= A.linear_dynamics(x0,M,t_end)
    push!(xs_all,xs)
    push!(t_all,t)
  end
  t_all,xs_all
end

function norm_ribbons(mats)
  nrms = map(mats) do mat
    mapslices(norm,mat;dims=1)
  end
  nrms = vcat(nrms...)
  means = mean(nrms;dims=1)[:]
  lowhigh = mapslices(m->quantile(m,[0.05,0.95]),nrms;dims=1)
  low=lowhigh[1,:][:]
  high=lowhigh[2,:][:]
  return (means,low,high)
end

##
# start with chaotic!
n=10
M = randn(n,n) + (0.3333I)
myheatmap(M)

x0 = randn(n)
@time t_one,xs_one = A.linear_dynamics_dumb(x0,M,0.5)
@time t_one,xs_two = A.linear_dynamics(x0,M,0.5)

plot(t_one, xs_one[1,:])
scatter!(t_one, xs_two[1,:] ; marker=:circle)


t_all,xs_all = klinrun(50,1.0,M,1.0)

_ = let norms = norm_ribbons(xs_all)
  plot(t_all[1],norms[1] ; linewidth=4,
    leg=false, ribbon=(norms[2],norms[3]),color=:green)
end

##


x0 = randn(n)
t,xs = A.linear_dynamics(x0,M,10.0)
A.run_network
A.run_network_withnoise
A.run_network_to_convergence  

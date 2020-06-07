
using Pkg
Pkg.activate(joinpath(@__DIR__(),".."))
using Plots ; theme(:dark)
using AttractorNetworksTests ; const A=AttractorNetworksTests
using LinearAlgebra
using Random ; Random.seed!(0)
using Test

##
# linear ntw should produce perfectly linear dynamics
# define matrix
ne=7
ni=22
n=ne+ni
M = randn(n,n)
ntw = A.RecurrentNetwork(ne,ni ; gfun=A.B.GFId(), taus=ones(n),
    external_current=zeros(n))
copyto!(ntw.weights,M+I)
# now run the linear dynamics
t_end,stepsize =1.0,0.025
x0 = 0.1 .* randn(n)
t_out1,dyn1 = A.linear_dynamics(x0,M,t_end;verbose=true,stepsize=stepsize)
# and run the network
t_out2,dyn2 = A.run_network(x0,t_end,ntw;verbose=true,stepsize=stepsize)
@test all(t_out1 .== t_out2)
@test all( isapprox.(dyn1,dyn2 ; atol=1E-3))

dyn1
dyn2
plot()

plot(t_out1,[dyn1[1,:]  dyn2[1,:]] ;
    leg=false, opacity=0.5 , linewidth=3)
@test true

A.B.velocity(x0,ntw) - (M*x0)

ntw.weights - M

extrema(dyn1 .- dyn2)

##

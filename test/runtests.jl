using AttractorNetworksTests; const A= AttractorNetworksTests
using Random ; Random.seed!(0)
using LinearAlgebra, Statistics
using Test

@testset "Linear dynamics" begin
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
end

@testset "Convergence to attractor" begin
    ne=7
    ni=22
    n=ne+ni
    # stable dynamics, diagonal weights
    wdiag = 0.5*rand(n)
    M = diagm(0=>wdiag)
    currs = 2.0randn(n)
    ntw = A.RecurrentNetwork(ne,ni ; gfun=A.B.GFId(), taus=1 .+ rand(n),
        external_current=currs)
    copyto!(ntw.weights,M)
    attr_an = @. -currs/(wdiag-1)
    x0=zeros(n)
    _,attr_num = A.run_network_to_convergence(x0,ntw)
    @test all(isapprox.(attr_an,attr_num ; rtol=1E-3 ))
end

@testset "Dynamics with noise" begin
    ne=7
    ni=22
    n=ne+ni
    # stable dynamics, diagonal weights
    wdiag = 0.5*rand(n)
    M = diagm(0=>wdiag)
    currs = 10.0randn(n)
    ntw = A.RecurrentNetwork(ne,ni ; gfun=A.B.GFId(), taus=1 .+ rand(n),
        external_current=currs)
    copyto!(ntw.weights,M)
    attr_an = @. -currs/(wdiag-1)
    u0=attr_an
    noise_lev1 = 3.0
    t,u_noise=A.run_network_withnoise(u0,80,noise_lev1,ntw; stepsize=0.1)
    cov1 = cov(u_noise;dims=2)
    @test isapprox(sqrt(mean(diag(cov1))),noise_lev1 ; rtol=0.2)
    noise_lev2 = 11.0
    t,u_noise=A.run_network_withnoise(u0,80,noise_lev2,ntw; stepsize=0.1)
    cov2 = cov(u_noise;dims=2)
    @test isapprox( sqrt(mean(diag(cov2))),noise_lev2 ; rtol=0.2)
end

@testset "Convergence to effective attractor" begin
    ne=7
    ni=22
    n=ne+ni
    # stable dynamics, diagonal weights
    wdiag = 0.5*rand(n)
    M = diagm(0=>wdiag)
    currs = 10.0randn(n)
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
end

module AttractorNetworksTests
using DifferentialEquations
using LinearAlgebra
using AttractorNetworksBase ; const B=AttractorNetworksBase

using JLD

# load and import
const RecurrentNetwork = B.RecurrentNetwork
const velocity! = B.velocity!
const velocity = B.velocity

# not analytic, but much faster... meh
function linear_dynamics_ode(x0::AbstractVector,A::AbstractMatrix,t_end::Real;
        verbose=true,stepsize=0.02)
    ode_solver = Tsit5()
    f(du,u,p,t) =  mul!(du,A,u)
    prob = ODEProblem(f,x0,(0.,t_end))
    solv =  solve(prob,ode_solver;verbose=verbose , saveat=stepsize)
    return solv.t,hcat(solv.u...)
end

function linear_dynamics(x0::AbstractVector{T},A::AbstractMatrix,t_end::Real;
        verbose=true,stepsize=0.02) where T
    ts = collect(0:stepsize:t_end)
    n = length(x0)
    nts = length(ts)
    ret = Matrix{T}(undef,n,nts)
    for (t,tt) in enumerate(ts)
        ret[:,t]=exp(A .*tt )*x0
    end
    return ts,ret
end

"""

"""
function run_network(x0::AbstractVector,t_max,rn::RecurrentNetwork;
            verbose::Bool=false , stepsize=0.05)
    ode_solver = Tsit5()
    gu_alloc = similar(x0)
    f(du,u,p,t) = velocity!(du,u,B.g!(gu_alloc,u,rn.gain_function),rn)
    prob = ODEProblem(f,x0,(0.,t_max))
    solv =  solve(prob,ode_solver;verbose=verbose,saveat=stepsize)
    return solv.t,hcat(solv.u...)
end


"""
    run_network_to_convergence(u0, rn::RecurrentNetwork ;
            t_max=50. , veltol::Float64=1E-4)

# inputs
  + `u0` : starting point, in terms of membrane potential
  + `rn` : network
  + `t_max` : maximum time considered (in seconds)
  + `veltol` : tolerance for small velocity
Runs the network stopping when the norm of the velocity is below
the tolerance level (stationary point reached)
"""
function run_network_to_convergence(u0, rn::RecurrentNetwork ;
        t_max=50. , veltol::Float64=1E-4)
    n=length(u0) |> Float64
    function  condition(u,t,integrator)
        v = get_du(integrator)
        return norm(v) / n < veltol
    end
    function affect!(integrator)
        savevalues!(integrator)
        return terminate!(integrator)
    end
    cb=DiscreteCallback(condition,affect!)
    ode_solver = Tsit5()
    f(du,u,p,t) = velocity!(du,u,rn)
    prob = ODEProblem(f,u0,(0.,t_max))
    out = solve(prob,Tsit5();verbose=false,callback=cb)
    u_out = out.u[end]
    t_out = out.t[end]
    if isapprox(t_out,t_max; atol=0.05)
        @warn "no convergence after max time $t_max"
        vel = velocity(u_out,rn)
        @info "the norm (divided by n) of the velocity is $(norm(vel)/n) "
    end
    return u_out
end


# check the status of the optimization
function read_optimization_costs(dir,fnamebase)
    @assert isdir(dir) "save directory $dir not found!"
    allfiles = readdir(dir)
    # filter the files
    goodfiles = filter(
        f->occursin(Regex("\\b\\d{3,}._$(fnamebase)"),f) , readdir(dir) )
    ret_iters= []
    ret_times = []
    ret_costs = []
    ret_xs = []
    for file in goodfiles
        jldopen(joinpath(dir,file),"r") do f
            push!(ret_iters,read(f,"iters"))
            push!(ret_times,read(f,"times"))
            push!(ret_costs,read(f,"costs"))
            push!(ret_xs,read(f,"xs"))
        end
    end
    return (iter = vcat(ret_iters...) , time = vcat(ret_times...) ,
     costs = vcat(ret_costs...) , xs = vcat(ret_xs...) )
end


end # module

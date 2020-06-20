module AttractorNetworksTests
using DifferentialEquations
using LinearAlgebra
using AttractorNetworksBase ; const B=AttractorNetworksBase

using JLD

# load and import
const RecurrentNetwork = B.RecurrentNetwork
const velocity! = B.velocity!
const velocity = B.velocity


function linear_dynamics_ode(x0::AbstractVector,A::AbstractMatrix,t_end::Real;
        verbose=true,stepsize=0.02)
    ode_solver = Tsit5()
    f(du,u,p,t) =  mul!(du,A,u)
    prob = ODEProblem(f,x0,(0.,t_end))
    solv =  solve(prob,ode_solver;verbose=verbose , saveat=stepsize)
    return solv.t,hcat(solv.u...)
end

@doc raw"""
        linear_dynamics(x0,A,t_end;stepsize=0.02)

Solves analytically the following differential equation
```math
\mathrm d \mathbf x / \mathrm d t = A \; \mathbf x
```
for `t=(0:stepsize:t_end)`

# Arguments
- `x0::Vector`
- `A::Matrix`
- `t_end::Real`: the final timepoint
- `stepsize::Real`
"""
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

@doc raw"""
        run_network(x0::AbstractVector,t_end,rn::RecurrentNetwork;
            verbose::Bool=false , stepsize=0.05)

Solves the following differential equation
```math
\frac{\mathrm d \mathbf u_i} {\mathrm d t} = (1/\tau_i)\left(
 -u_i + h_i + \sum_{ij} W_{ij}\,g(u_j)
 \right)
```
for `t=(0:stepsize:t_end)`, where all elements are in the `rn`
object. Returns `(t,u_t)`

# Arguments
- `u0::Vector` : initial conditions
- `rn::RecurrentNetwork`
- `t_end::Real`: the final timepoint
- `stepsize::Real`
- `verbose::Bool` : passed to the ODE `solve` function

# Outputs
- `t::Vector` : the time steps
- `u_t::Matrix` : each column is the corresponding $u(t)$
"""
function run_network(u0::AbstractVector,t_end,rn::RecurrentNetwork;
            verbose::Bool=false , stepsize=0.05)
    @assert length(u0)==B.n_neurons(rn)
    ode_solver = Tsit5()
    gu_alloc = similar(u0)
    f(du,u,p,t) = velocity!(du,u,B.g!(gu_alloc,u,rn.gain_function),rn)
    prob = ODEProblem(f,u0,(0.,t_end))
    solv =  solve(prob,ode_solver;verbose=verbose,saveat=stepsize)
    return solv.t,hcat(solv.u...)
end

"""
        run_network_withnoise(u0::AbstractVector,t_end,noiselevel,
                        rn::RecurrentNetwork;
                        verbose::Bool=false , stepsize=0.05)

See documentation of [`run_network`](@ref). This function is equivalent,
except a diagonal, gaussian noise term is added to the dynamics.
The noise level would be approximately the std around the attractor in
linear dynamics.
"""
function run_network_withnoise(u0::AbstractVector,t_end,noiselevel,
                rn::RecurrentNetwork;
                verbose::Bool=false , stepsize=0.05)
    @assert length(u0)==B.n_neurons(rn)
    ode_solver = Tsit5()
    gu_alloc = similar(u0)
    f(du,u,p,t) = velocity!(du,u,B.g!(gu_alloc,u,rn.gain_function),rn)
    σ_f(du,u,p,t) = fill!(du,noiselevel)
    prob = SDEProblem(f,σ_f,u0,(0.,t_end))
    solv =  solve(prob,ode_solver;verbose=verbose,saveat=stepsize)
    return solv.t,hcat(solv.u...)
end

"""
        run_network_to_convergence(u0, rn::RecurrentNetwork ;
                t_end=80. , veltol=1E-4)

Runs the network as described in [`run_network`](@ref), but stops as soon as
`norm(v) / n < veltol` where `v` is the velocity at time `t`.
If this condition is not satisfied (no convergence to attractor), it runs until `t_end` and prints a warning.

# Arguments
- `u0::Vector` : initial conditions
- `rn::RecurrentNetwork`
- `t_end::Real`: the maximum time considered
- `veltol::Real` : the norm (divided by num. dimensions) for velocity at convergence
# Outputs
- `u_end`::Vector : the final state at convergence
- `t_end`::Real : the corresponding time
"""
function run_network_to_convergence(u0::AbstractVector, rn::RecurrentNetwork ;
        t_end=80. , veltol::Float64=1E-4)
    n=length(u0) |> Float64
    @assert n==B.n_neurons(rn)
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
    gu_alloc = similar(u0)
    f(du,u,p,t) = velocity!(du,u,B.g!(gu_alloc,u,rn.gain_function),rn)
    prob = ODEProblem(f,u0,(0.,t_end))
    out = solve(prob,Tsit5();verbose=false,callback=cb)
    u_out = out.u[end]
    t_out = out.t[end]
    if isapprox(t_out,t_end; atol=0.05)
        vel = velocity(u_out,rn)
        @warn "no convergence after max time $t_end"
        @info "the norm (divided by n) of the velocity is $(norm(vel)/n) "
    end
    return t_out,u_out
end

# the distance between state space points can me measured in
# multiple ways
abstract type AttractorError end
Base.Broadcast.broadcastable(ae::AttractorError)=Ref(ae)
function (ae::AttractorError)(u,attr_u,ntw::RecurrentNetwork)
    g=ntw.gain_function
    ae(g.(u),g.(attr_u))
end

struct NormError <: AttractorError end
struct NormErrorPri <: AttractorError
    idx_pri
end

# signature should be (f::ftype)(r,attr_r)
# (f::ftype)(u,attr_u,ntw)

function (ae::NormError)(r,attr_r)
    ret = 0.0
    n = length(r)
    for (r1,r2) in zip(r,attr_r)
        ret += (r1-r2)^2
    end
    return ret/n
end

function effective_attractors(u_attr::Matrix,ntw::RecurrentNetwork;
           t_end=80. , veltol::Float64=1E-4,
           attr_err::AttractorError=NormError())
    ndims,nattr=size(u_attr)
    @assert ndims == B.n_neurons(ntw)
    u_converged = mapslices(u->run_network_to_convergence(u, ntw;
            t_end=t_end , veltol=veltol)[2] , u_attr ; dims=1)
    errs = map(1:nattr) do k
        attr_err(view(u_converged,:,k),view(u_attr,:,k),ntw)
    end
    return u_converged, errs
end


## I/O functions (during and after optimization)

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

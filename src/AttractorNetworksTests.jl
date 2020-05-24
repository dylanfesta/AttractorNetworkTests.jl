module AttractorNetworksTests


#=

These should go somewhere else, to avoid the heavy dependency with
DifferentialEquations !!!

"""
    run_network(x0,t_max,rn::RecurrentNetwork; rungekutta=false,verbose=false)

"""
function run_network(x0::AbstractVector,t_max,rn::RecurrentNetwork;
            verbose::Bool=false)
    ode_solver = Tsit5()
    f(du,u,p,t) = velocity!(du,u,rn)
    prob = ODEProblem(f,x0,(0.,t_max))
    return solve(prob,ode_solver;verbose=verbose)
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
=#

end # module

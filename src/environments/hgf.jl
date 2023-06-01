export HGFEnvironment

using Random

Base.@kwdef struct HGFEnvironment
    kappa::Float64 = 1.0
    omega::Float64 = 0.0
    layer1_variance::Float64 = abs2(0.01)
    noise::Float64 = abs2(0.25)
end

Base.show(io::IO, environment::HGFEnvironment) = print(io, "HGFEnvironment()")

function Random.rand(rng::AbstractRNG, environment::HGFEnvironment, T::Int)
    
    z_prev = 0.0
    x_prev = 0.0

    z = Vector{Float64}(undef, T)
    x = Vector{Float64}(undef, T)
    observations = Vector{Float64}(undef, T)

    @inbounds for i in 1:T
        z[i] = rand(rng, Normal(z_prev, sqrt(environment.layer1_variance)))
        x[i] = rand(rng, Normal(x_prev, sqrt(exp(environment.kappa * z[i] + environment.omega))))
        observations[i] = rand(rng, Normal(x[i], sqrt(environment.noise)))

        z_prev = z[i]
        x_prev = x[i]
    end 
    
    return z, x, observations
end

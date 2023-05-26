### Original code is written by Tim Nisslbeck
### Adapted by Dmitry Bagaev

using RxInfer, StaticArrays, Plots, StableRNGs, LinearAlgebra, Random

import Base: rand

export DoublePendulum, polar2cart_com, polar2cart_rod, polar2cart

"""
An environment for a double pendulum. 
Use the `state_transition` function to generate the efficient version of the state transition discretization.
"""
Base.@kwdef struct DoublePendulum
    Ja::Float64 = 1.3333333333333333
    Jb::Float64 = 0.3333333333333333
    Jx::Float64 = 0.5
    μ1::Float64 = 14.715
    μ2::Float64 = 4.905
    kt::Float64 = 0.0
    Δt::Float64 = 0.01
    γ::Float64 = 3.0
end

Base.show(io::IO, ::DoublePendulum) = print(io, "DoublePendulum()")

# State transition function

function state_transition(environment::DoublePendulum)

    return let environment = environment
        (s) -> begin

            Ja = environment.Ja
            Jb = environment.Jb
            Jx = environment.Jx
            μ1 = environment.μ1
            μ2 = environment.μ2
            kt = environment.kt
            Δt = environment.Δt

            # t
            K1 = double_pendulum_dzdt(s, Ja, Jb, Jx, μ1, μ2, kt)
            # t+dt/2
            K2 = double_pendulum_dzdt(s + K1 * Δt / 2, Ja, Jb, Jx, μ1, μ2, kt)
            # t+dt/2
            K3 = double_pendulum_dzdt(s + K2 * Δt / 2, Ja, Jb, Jx, μ1, μ2, kt)
            # t+dt
            K4 = double_pendulum_dzdt(s + K3 * Δt, Ja, Jb, Jx, μ1, μ2, kt)

            Δs = Δt * 1 / 6 * (K1 + 2 * K2 + 2 * K3 + K4)

            nexts = s + Δs

            @inbounds SA[nexts[1]%2π, nexts[2]%2π, nexts[3], nexts[4]]
        end
    end

end

function double_pendulum_dzdt(z, Ja, Jb, Jx, μ1, μ2, kt)
    return @inbounds begin
        z1mz2 = z[1] - z[2]
        sinz1mz2 = sin(z1mz2)
        z2mz1 = z[2] - z[1]
        sinz2mz1 = sin(z2mz1)

        # Shorthand for matrix inversion
        A = Ja
        B = Jx * cos(z1mz2)
        C = B
        D = Jb

        # Equations of motion: θ̈₁
        ddθ1 = -Jx * sinz1mz2 * z[4]^2 - μ1 * sin(z[1]) + kt * sinz2mz1
        ddθ2 = Jx * sinz1mz2 * z[3]^2 - μ2 * sin(z[2]) + kt * sinz2mz1

        # Inverse mass (inertia) matrix
        Mi = 1 / (A * D - B * C) * SA[D -B; -C A]

        ddθ = Mi * SA[ddθ1, ddθ2]

        SA[z[3]; z[4]; ddθ[1]; ddθ[2]]
    end
end

### Data generation related ###

function Base.rand(environment::DoublePendulum, T::Int)
    return rand(StableRNG(123), environment, T)
end

function Base.rand(rng::AbstractRNG, environment::DoublePendulum, T::Int; c = SA[0.0, 1.0, 0.0, 0.0], random_start = false)
    state_current = if !random_start
        SA[1.2, 0.2, 0.0, 0.0]
    else
        SA[ 0.5randn(rng), 0.5randn(rng), 0.0, 0.0 ]
    end
    states = Vector{typeof(state_current)}(undef, T)
    observations = Vector{Float64}(undef, T)
    states[1] = state_current
    transition = state_transition(environment)
    for t in 2:T
        @inbounds states[t] = rand(rng, MvNormalMeanPrecision(transition(states[t - 1]), 1e6I(4)))
    end
    observations = rand.(rng, (NormalMeanPrecision(dot(c, state), environment.γ) for state in states))
    return states, observations
end

### Plotting related ###

# Map angles to the cartesian space
# Assumes that lengths are equal to `1`
function polar2cart(state)

    # Position of first mass
    x1 = sin(state[1])
    y1 = -cos(state[1])

    # Position of second mass
    x2 = x1 + sin(state[2])
    y2 = y1 - cos(state[2])

    return (x1, y1), (x2, y2)
end

# Map angles of centers of masses to cartesian space
# Assumes that lengths are equal to `1`
function polar2cart_com(state)
    "Map angles of centers of masses to Cartesian space"

    # Position of first mass
    x1 = sin(state[1])
    y1 = -cos(state[1])

    # Position of second mass
    x2 = sin(state[1]) + sin(state[2])
    y2 = -cos(state[1]) - cos(state[2])

    return (x1, y1), (x2, y2)
end

# Map the end points of a rod to cartesian space
# Assumes that lengths are equal to `1`
function polar2cart_rod(state)

    # End position of first rod
    x1 = sin(state[1])
    y1 = -cos(state[1])

    # End position of second rod
    x2 = x1 + sin(state[2])
    y2 = y1 - cos(state[2])

    return (x1, y1), (x2, y2)
end

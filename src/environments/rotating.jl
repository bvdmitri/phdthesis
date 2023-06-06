
export RotatingTracking

using Random, Distributions, StableRNGs

"""
This environment represents a simple linear dynamical system where the state-transition
matrix is defined as a rotation matrix with some predetermined angle, such that 

    s ~ MvNormal(A * s_prev, P)
    y ~ MvNormal(B * s, Q)

- `d` - represents dimension of the state vector
- `A` - state-transition matrix
- `B` - observational matrix
- `P` - state-transition noise
- `Q` - observational noise
"""
struct RotatingTracking 
    d::Int
    A::Matrix
    B::Matrix
    P::Matrix
    Q::Matrix
end

# At this point we simply generate a random rotation state-transition matrix
# Noise matrices are fixed
function RotatingTracking(d::Int = 2; rng = StableRNG(123))
    A = random_rotation_matrix(rng, d)
    B = Matrix(Diagonal(ones(d) .+ rand(rng, -0.5:0.1:1.0, d)))
    P = Matrix(Diagonal(2.0 * ones(d)))
    Q = Matrix(Diagonal(2.0 * ones(d)))
    return RotatingTracking(d, A, B, P, Q)
end

function Random.rand(rng::AbstractRNG, environment::RotatingTracking, T::Int)

    d = environment.d
    A = environment.A
    B = environment.B
    P = environment.P
    Q = environment.Q

    @assert size(A) === (d, d)
    @assert size(B) === (d, d)
    @assert size(P) === (d, d)
    @assert size(Q) === (d, d)

    s_prev = ones(d)

    s = Vector{Vector{Float64}}(undef, T)
    y = Vector{Vector{Float64}}(undef, T)

    for i in 1:T
        s[i] = rand(rng, MvNormal(A * s_prev, P))
        y[i] = rand(rng, MvNormal(B * s[i], Q))

        s_prev = s[i]
    end
   
    return s, y
end

function random_rotation_matrix(rng, dimension)
    R = Matrix(Diagonal(ones(dimension)))

    θ = π/4 * rand(rng)

    for i in 1:dimension 
        for j in (i + 1):dimension
            S = Matrix(Diagonal(ones(dimension)))
            S[i, i] = cos(θ)
            S[j, j] = cos(θ)
            S[i, j] = sin(θ)
            S[j, i] = -sin(θ)
            R = R * S
        end
    end
    return R
end


export RotatingTracking

using Random, Distributions

struct RotatingTracking 
    dimension::Int
    angle::Float64
    A::Matrix
    B::Matrix
    P::Matrix
    Q::Matrix
end

function RotatingTracking(dimension::Int = 2, angle::Float64 = π / 20)
    A = rotation_matrix(angle)
    B = diagonal_matrix([ 1.3, 0.7 ])
    P = diagonal_matrix([ 1.0, 1.0 ])
    Q = diagonal_matrix([ 1.0, 1.0 ])

    return RotatingTracking(
        dimension,
        angle,
        A,
        B,
        P,
        Q
    )
end

function Random.rand(rng::AbstractRNG, environment::RotatingTracking, T::Int)

    d = environment.dimension
    A = environment.A
    B = environment.B
    P = environment.P
    Q = environment.Q

    @assert size(A) === (d, d)
    @assert size(B) === (d, d)
    @assert size(P) === (d, d)
    @assert size(Q) === (d, d)

    x_prev = zeros(d)

    x = Vector{Vector{Float64}}(undef, T)
    y = Vector{Vector{Float64}}(undef, T)

    for i in 1:T
        x[i] = rand(rng, MvNormal(A * x_prev, P))
        y[i] = rand(rng, MvNormal(B * x[i], Q))

        x_prev = x[i]
    end
   
    return x, y
end

function rotation_matrix(θ)
    return [ 
        cos(θ) -sin(θ); 
        sin(θ) cos(θ) 
    ]
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

function diagonal_matrix(values)
    return Matrix(Diagonal(values))
end

function random_posdef_matrix(rng, dimension)
    L = rand(rng, dimension, dimension)
	return L' * L
end

function random_vector(rng, distribution::Distributions.Categorical) 
    k = ncategories(distribution)
    s = zeros(k)
    s[ rand(rng, distribution) ] = 1.0
    s
end

function normalise(a)
	return a ./ sum(a)
end

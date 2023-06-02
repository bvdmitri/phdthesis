# NOTE: this file is not included in the project by default, 
# it must be included explicitly in the notebook experiments

import ReactiveMP

using ReverseDiff

# Turing is more efficient with ReverseDiff for this model
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Hot fixes for `ReverseDiff`
Base.:(+)(a::ReverseDiff.TrackedArray, b::SVector) = +(a, convert(Vector, b))

@model rotating(y, A, B, P, Q, ::Type{TV} = Vector{Vector{Float64}}) where TV = begin
    n = length(y)

    # State sequence.
    x = TV(undef, n)

    d  = size(A, 1)
    pm = zeros(d)
    pc = Matrix(Diagonal(100.0 * ones(d)))

    # Observe each point of the input.
    x[1] ~ MvNormal(pm, pc)
    y[1] ~ MvNormal(B * x[1], Q)

    for t in 2:n
        x[t] ~ MvNormal(A * x[t - 1], P)
        y[t] ~ MvNormal(B * x[t], Q)
    end
end

function run_inference_sampling(model; method = NUTS(), nsamples = 1000, rng = StableRNG(123))
    return sample(rng, model, method, nsamples)
end

reshape_turing_data(d, data) = transpose(reshape(data, (d, Int(length(data) / d))))

function extract_posteriors(T, d, results)
    sposteriors = get(results, :x)
    em = reshape_turing_data(d, [ mean(sposteriors.x[i].data) for i in 1:(d * T) ]) |> collect |> eachrow |> collect
    ev = reshape_turing_data(d, [ var(sposteriors.x[i].data) for i in 1:(d * T) ]) |> collect |> eachrow |> collect;
    
    return map(tuple -> ReactiveMP.MvNormalMeanCovariance(tuple[1], Diagonal(tuple[2])), zip(em, ev))
end

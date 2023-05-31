# NOTE: this file is not included in the project by default, 
# it must be included explicitly in the notebook experiments

import ReactiveMP

# Turing is more efficient with ReverseDiff for this model
Turing.setadbackend(:reversediff)
Turing.setrdcache(true)

# Hot fixes for `ReverseDiff`
Base.:(+)(a::ReverseDiff.TrackedArray, b::SVector) = +(a, convert(Vector, b))

@model function double_pendulum(y, T, ::Type{TV} = Vector{Vector{Float64}}) where TV
    
    # State sequence.
    s = TV(undef, T)
    σ ~ Gamma(0.001, 100.0)

    # This line is needed for VI, which may optimize out of the domain
    σ = abs(σ)
    
    Σ = inv(1e4) * I(4)
    c = [ 0.0, 1.0, 0.0, 0.0 ]
    
    s[1] ~ MvNormal(zeros(4), I(4))
    y[1] ~ Normal(dot(s[1], c), sqrt(inv(σ)))
    
    for t in 2:T
        s[t] ~ MvNormal(f(s[t - 1]), Σ)
        y[t] ~ Normal(dot(s[t], c), sqrt(inv(σ)))
    end
end

function sample_inference(tgmodel; method = NUTS(), nsamples = 1000, rng = StableRNG(42))
    return sample(rng, tgmodel, method, nsamples)
end

reshape_data(data) = transpose(reduce(hcat, data))
reshape_turing_data(data) = transpose(reshape(data, (4, Int(length(data) / 4))))

function extract_posteriors(T, results)
    sposteriors = get(results, :s)
    em = reshape_turing_data([ mean(sposteriors.s[i].data) for i in 1:4T ]) |> collect |> eachrow |> collect
    ev = reshape_turing_data([ var(sposteriors.s[i].data) for i in 1:4T ]) |> collect |> eachrow |> collect;
    
    return map(tuple -> ReactiveMP.MvNormalMeanCovariance(tuple[1], Diagonal(tuple[2])), zip(em, ev))
end

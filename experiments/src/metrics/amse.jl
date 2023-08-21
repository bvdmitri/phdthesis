import RxInfer, ForneyLab, Turing

export compute_amse

function compute_amse end

function compute_amse(states::AbstractVector, estimated::AbstractVector)
    return compute_amse(eltype(states), eltype(estimated), states, estimated)
end

function compute_amse(::Type{T}, ::Type{ Any }, states, estimated) where { T }
    return compute_amse(T, typeof(first(estimated)), states, estimated)
end

## ReactiveMP generic 

function compute_amse(::Type{T}, ::Type{ <: ReactiveMP.Marginal }, states, estimated) where { T }
    return compute_amse(T, typeof(ReactiveMP.getdata(first(estimated))), states, estimated)
end

## ForneyLab generic

function compute_amse(::Type{T}, ::Type{ ForneyLab.ProbabilityDistribution }, states, estimated) where { T }
    return compute_amse(T, typeof(first(estimated)), states, estimated)
end

function compute_amse(::Type{T}, ::Type{ <: ForneyLab.ProbabilityDistribution{V, F} }, states, estimated) where { T, V, F }
    return compute_amse(T, F, states, estimated)
end

## RxInfer

function compute_amse(::Type{ <: Real }, ::Type{ <: ReactiveMP.UnivariateGaussianDistributionsFamily }, states, estimated)
    return 1.0 / length(states) * mapreduce(+, zip(states, estimated)) do (s, e)
        return var(e) + abs2(s - mean(e))
    end
end

function compute_amse(::Type{ <: AbstractVector }, ::Type{ <: ReactiveMP.MultivariateGaussianDistributionsFamily }, states, estimated)
    return 1.0 / length(states) * mapreduce(+, zip(states, estimated)) do (s, e)
        diff = s .- mean(e)
        return tr(cov(e)) + diff' * diff 
    end
end

## ForneyLab

function compute_amse(::Type{ <: AbstractVector }, ::Type{ <: ForneyLab.Gaussian }, states, estimates)
    converted = map(estimates) do e 
        return ReactiveMP.MvNormalMeanCovariance(ForneyLab.unsafeMeanCov(e)...)
    end
    return compute_amse(states, converted)
end

function compute_amse(::Type{ <: Real }, ::Type{ <: ForneyLab.Gaussian }, states, estimates)
    converted = map(estimates) do e 
        return ReactiveMP.NormalMeanVariance(ForneyLab.unsafeMeanCov(e)...)
    end
    return compute_amse(states, converted)
end


## Turing

function compute_amse(states::AbstractVector, chains::Turing.Chains, s::Symbol, ::Type{ MvNormal })
    d = length(first(states))
    
    reshape_turing_data = (data) -> transpose(reshape(data, (d, Int(length(data) / d))))

    n_turing = length(states)
    samples  = get(chains, s)
    
    means = reshape_turing_data([ mean(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect
    covs = reshape_turing_data([ var(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect

    estimated = map(e -> MvNormal(vec(e[1]), vec(e[2])), zip(means, covs))
    
    return 1.0 / length(states) * mapreduce(+, zip(states, estimated)) do (s, e)
        diff = s .- mean(e)
        return tr(cov(e)) + diff' * diff 
    end
end

function compute_amse(states::AbstractVector, chains::Turing.Chains, s::Symbol, ::Type{ Normal })
    d = length(first(states))
    
    reshape_turing_data = (data) -> transpose(reshape(data, (d, Int(length(data) / d))))

    n_turing = length(states)
    samples  = get(chains, s)
    
    means = reshape_turing_data([ mean(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect
    stds = reshape_turing_data([ std(getfield(samples, s)[i].data) for i in 1:(d * n_turing) ]) |> collect |> eachrow |> collect

    estimated = map(e -> Normal(e[1], e[2]), zip(means, stds))
    
    return 1.0 / length(states) * mapreduce(+, zip(states, estimated)) do (s, e)
        diff = s .- mean(e)
        return var(e) + diff' * diff 
    end
end

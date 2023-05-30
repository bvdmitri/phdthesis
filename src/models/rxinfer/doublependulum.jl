# NOTE: this file is not included in the project by default, 
# it must be included explicitly in the notebook experiments

# Hot fixes for `ReactiveMP`
# These fixes are needed for the `StaticArray` library, this should be eventually 
# ported to the main code of the `ReactiveMP`, but I decided to keep it here for now
ReactiveMP.cholinv(x::UniformScaling) = inv(x.λ) * I
ReactiveMP.cholinv(x::SMatrix) = inv(x)
ReactiveMP.fastcholesky(x::SMatrix) = ReactiveMP.fastcholesky(Matrix(x))

# Slightly optimized version of the model used in the thesis
# - state + N(0, precision) is replaced with N(state, precision) as they are identical, but generate less nodes
# - arrays are replaced with their static array equivalents with the SA prefix
# - variable names are different, but that is minor IMO
@model function double_pendulum(T)
    
    s = randomvar(T)
    y = datavar(Float64, T)
    σ ~ Gamma(shape = 0.001, rate = 0.01)
    
    Σ = constvar(SMatrix{4, 4}(1e4 * diageye(4)))
    c = constvar(SA[ 0.0, 1.0, 0.0, 0.0 ])
    
    s[1] ~ MvNormal(mean = zeros(4), covariance = SMatrix{4, 4}(diageye(4)))
    y[1] ~ Normal(mean = dot(s[1], c), precision = σ)
    
    for t in 2:T
        s[t] ~ MvNormal(mean = f(s[t - 1]), precision = Σ)
        y[t] ~ Normal(mean = dot(s[t], c), precision = σ)
    end
end

@meta function double_pendulum_meta()
    f() -> Linearization()
end

@constraints function double_pendulum_constraints()
    q(s, σ) = q(s)q(σ)
end

# First invoke slow as usual, but we do not perform benchmarks here so it does not really matter
# We perform a small benchmark below, for that we need to wrap the `inference` call in a separate function
function run_inference(rximodel, observations; iterations = 5)
    return inference(
        model = rximodel,
        data = (y = observations, ),
        meta = double_pendulum_meta(),
        constraints = double_pendulum_constraints(),
        returnvars = (s = KeepLast(), ),
        iterations = iterations,
        initmarginals = (σ = GammaShapeRate(0.001, 0.01), ),
        options = (limit_stack_depth = 500, )
    )
end

function extract_posteriors(T, results)
    @assert length(results.posteriors[:s]) === T
    return map(q -> MvNormalMeanCovariance(mean(q), cov(q)), results.posteriors[:s])
end

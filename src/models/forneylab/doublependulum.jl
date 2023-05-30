# NOTE: this file is not included in the project by default,
# it must be included explicitly in the notebook experiments

import ForneyLab: unsafeMean, unsafeCov
import Distributions, ReactiveMP

const flmodels = Dict()

function make_forneylab_model(T; id = "")
    # This creates new global variable for the current factor graph
    g = FactorGraph()
    
    @RV σ ~ Gamma(0.001, 0.01)

    # Pre-define vectors for storing latent and observed variables
    # To access the result we therefore need to define more latent variables
    h = Vector{Variable}(undef, T - 1) 
    u = Vector{Variable}(undef, T)
    s = Vector{Variable}(undef, T) 
    y = Vector{Variable}(undef, T)

    @RV s[1] ~ GaussianMeanPrecision(zeros(4), Matrix(I(4)))
    @RV u[1] = dot(s[1], [ 0.0, 1.0, 0.0, 0.0 ])
    @RV y[1] ~ GaussianMeanPrecision(u[1], σ)
    
    # Indicate observed variable
    placeholder(y[1], :y, index = 1)
    
    for t in 2:T
        @RV h[t - 1] ~ Delta{Extended}(s[t - 1]; g = f)
        @RV s[t] ~ GaussianMeanPrecision(h[t - 1], Matrix(1e4 * I(4)))
        @RV u[t] = dot(s[t], [ 0.0, 1.0, 0.0, 0.0 ])
        @RV y[t] ~ GaussianMeanPrecision(u[t], σ)

        # Indicate observed variable
        placeholder(y[t], :y, index = t) 
    end
    
    # q(σ, u) = q(σ)q(u)
    q = PosteriorFactorization(σ, u, ids=[:σ, :u])
    
    # Build the variational update algorithms for each posterior factor
    algo = messagePassingAlgorithm(id = Symbol(string("a", id)))

    # Generate source code for the algorithms
    source_code = algorithmSourceCode(algo)
    
    # ForneyLab returns generated code as a string
    parsed = Meta.parse(source_code)
    
    # ForneyLab evaluates the generated code dynamically
    eval(parsed)
    
    # ForneyLab creates global functions dynamically
    # That is a bit unfortunate as it does not play nicely if we want 
    # to perform benchmarks in a loop
    # Here we attempt to get the references for the generated functions dynamically
    # and return them separately
    steps = () -> begin
        stepσ = Base.getproperty(Main, Symbol(string("stepa", id, "σ!")))
        stepu = Base.getproperty(Main, Symbol(string("stepa", id, "u!")))
        initu = Base.getproperty(Main, Symbol(string("inita", id, "u")))
        return (
            (args...) -> Base.invokelatest(stepσ, args...), 
            (args...) -> Base.invokelatest(stepu, args...), 
            (args...) -> Base.invokelatest(initu, args...)
        )
    end
    
    return steps
end

# ForneyLab needs to compile the model first
function double_pendulum(T; force = false)
    id = string(T) # unique id
    # do not recompile the model if not needed
    if !force
        return get!(() -> make_forneylab_model(T, id = id), flmodels, id)
    else
        model = make_forneylab_model(T, id = id)
        flmodels[id] = model
        return model
    end
end

# First invoke slow as usual, but we do not perform benchmarks here so it does not really matter
# We perform a small benchmark below, for that we need to wrap the `inference` call in a separate function
function run_inference(flmodel, observations; iterations = 5)
    (stepσ!, stepu!, initu) = flmodel()
    
    data = Dict(
        :y => observations,
    )
    
    marginals = Dict{Any, Any}(
        :σ => ProbabilityDistribution(Gamma, a = 0.001, b = 0.01)
    )
    
    messages = initu()
    
    for vmp in 1:iterations
        stepu!(data, marginals, messages)
        stepσ!(data, marginals, messages)
    end
    
    return marginals
end

function extract_posteriors(T, results)
    sposteriors = []
    push!(sposteriors, (unsafeMean(results[:s_1]), unsafeCov(results[:s_1])))
    for t in 2:T
        sym = Symbol(string("s_", t, "_h_", t - 1))
        push!(sposteriors, (unsafeMean(results[sym])[1:4], unsafeCov(results[sym])[1:4, 1:4]))
    end
    return map(tuple -> ReactiveMP.MvNormalMeanCovariance(tuple[1], tuple[2]), sposteriors)
end

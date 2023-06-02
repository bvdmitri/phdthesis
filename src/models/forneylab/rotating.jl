# NOTE: this file is not included in the project by default,
# it must be included explicitly in the notebook experiments

import ForneyLab: unsafeMean, unsafeCov
import Distributions, ReactiveMP

const flmodels = Dict()

function make_forneylab_model(T, environment; id = "")
    g = FactorGraph()

    model_d = environment.d
    model_n = T
    model_A = environment.A
    model_B = environment.B
    model_P = environment.P
    model_Q = environment.Q

    @RV x0 ~ GaussianMeanVariance(zeros(model_d), Matrix(Diagonal(100.0 * ones(model_d)))) # Prior

    x = Vector{Variable}(undef, model_n) # Pre-define vectors for storing latent and observed variables
    y = Vector{Variable}(undef, model_n)

    x_t_prev = x0

    for t = 1:model_n
        @RV x[t] ~ GaussianMeanVariance(model_A*x_t_prev, model_P) # Process model
        @RV y[t] ~ GaussianMeanVariance(model_B*x[t], model_Q) # Observation model

        placeholder(y[t], :y, dims=(model_d,), index=t) # Indicate observed variable
        
        x_t_prev = x[t] # Prepare state for next section
    end


    # Build the variational update algorithms for each posterior factor
    algo = messagePassingAlgorithm(x; id = Symbol(string(id)))


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
    return () -> begin
        return (args...) -> Base.invokelatest(Base.getproperty(Main, Symbol(string("step", id, "!"))), args...)
    end
end

# ForneyLab needs to compile the model first
# We need seed in the model ID because we use fixed matrices
function rotating(T, seed, env; force = false)
    id = string(T, "_", seed, "_", env.d) # unique id
    # do not recompile the model if not needed
    if !force
        return get!(() -> make_forneylab_model(T, env, id = id), flmodels, id)
    else
        model = make_forneylab_model(T, env, id = id)
        flmodels[id] = model
        return model
    end
end


# First invoke slow as usual, but we do not perform benchmarks here so it does not really matter
# We perform a small benchmark below, for that we need to wrap the `inference` call in a separate function
function run_inference(flmodel, observations)
    step! = flmodel()
    
    data = Dict(
        :y => observations,
    )
    
    inferred = step!(data) # Execute inference
    marginals = map(i -> inferred[Symbol(:x_, i)], 1:length(observations)) 

    return marginals
end

function extract_posteriors(T, results)
    sposteriors = []
    for t in 1:T
        push!(sposteriors, (unsafeMean(results[t]), unsafeCov(results[t])))
    end
    return map(tuple -> ReactiveMP.MvNormalMeanCovariance(tuple[1], tuple[2]), sposteriors)
end

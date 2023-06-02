# NOTE: this file is not included in the project by default,
# it must be included explicitly in the notebook experiments

import ForneyLab: unsafeMean, unsafeCov
import Distributions, ReactiveMP

include(srcdir("models", "forneylab", "fl_gcv", "GCV.jl"))

import .GCV: ruleMGaussianControlledVarianceGGDDD, ruleMGaussianMeanPrecisionEGD
import .GCV: ruleSVBGaussianControlledVarianceOutNGDDD, ruleSVBGaussianMeanPrecisionMEND
import .GCV: ruleSVBGaussianControlledVarianceXGNDDD, ruleSVBGaussianControlledVarianceZDNDD
import .GCV: ruleSVBGaussianControlledVarianceXGNDDD
import .GCV: GaussianControlledVariance

const flmodels = Dict()

function make_forneylab_model(kappa, omega; id = "")
    g = FactorGraph()

    @RV zp_shape
    @RV zp_rate

    placeholder(zp_shape, :zp_shape)
    placeholder(zp_rate, :zp_rate)

    @RV yp_shape
    @RV yp_rate

    placeholder(yp_shape, :yp_shape)
    placeholder(yp_rate, :yp_rate)

    @RV zv_min_mean
    @RV zv_min_prec

    placeholder(zv_min_mean, :zv_min_mean)
    placeholder(zv_min_prec, :zv_min_prec)

    @RV xv_min_mean
    @RV xv_min_prec

    placeholder(xv_min_mean, :xv_min_mean)
    placeholder(xv_min_prec, :xv_min_prec)

    @RV zv_min ~ GaussianMeanPrecision(zv_min_mean, zv_min_prec)
    @RV xv_min ~ GaussianMeanPrecision(xv_min_mean, xv_min_prec)

    @RV zp ~ Gamma(zp_shape, zp_rate)

    @RV zv ~ GaussianMeanPrecision(zv_min, zp)
    @RV xv ~ GaussianControlledVariance(xv_min, zv, kappa, omega)

    @RV yp ~ Gamma(yp_shape, yp_rate)

    @RV yv ~ GaussianMeanPrecision(xv, yp)

    placeholder(yv, :yv)
    
    # q(yp, xt, zt, xt_min) = q(xt, xt_min)q(zt)q(yp)
    # q(zt, zt_min, zp) = q(zt, zt_min)q(zp)
    pfz = PosteriorFactorization(zp, yp, [ zv, zv_min ], [xv, xv_min], ids=[ :ZP, :YP, :Z, :X ])
    
    # Build the variational update algorithms for each posterior factor
    algo = messagePassingAlgorithm(free_energy = true, id = Symbol(string(id))) 
    
    # Generate source code for the algorithms
    source_code = algorithmSourceCode(algo, free_energy = true); 
    
    # ForneyLab returns generated code as a string
    parsed = Meta.parse(source_code)
    
    # ForneyLab evaluates the generated code dynamically
    eval(parsed); 
    
    # ForneyLab creates global functions dynamically
    # That is a bit unfortunate as it does not play nicely if we want 
    # to perform benchmarks in a loop
    # Here we attempt to get the references for the generated functions dynamically
    # and return them separately
    steps = () -> begin
        stepX = Base.getproperty(Main, Symbol(string("step", id, "X!")))
        stepZ = Base.getproperty(Main, Symbol(string("step", id, "Z!")))
        stepZP = Base.getproperty(Main, Symbol(string("step", id, "ZP!")))
        stepYP = Base.getproperty(Main, Symbol(string("step", id, "YP!")))
        return (
            (args...) -> Base.invokelatest(stepX, args...), 
            (args...) -> Base.invokelatest(stepZ, args...), 
            (args...) -> Base.invokelatest(stepZP, args...),
            (args...) -> Base.invokelatest(stepYP, args...)
        )
    end
    
    return steps
end

# ForneyLab needs to compile the model first
# TODO: add kappa and omega to the ID
function hgf(kappa, omega; id = "", force = false)
    id = string(id) # unique id
    # do not recompile the model if not needed
    if !force
        return get!(() -> make_forneylab_model(kappa, omega; id = id), flmodels, id)
    else
        model = make_forneylab_model(kappa, omega; id = id)
        flmodels[id] = model
        return model
    end
end

function run_inference(model, observations; iterations = 5, showprogress = false)   
    stepX!, stepZ!, stepZP!, stepYP! = model()
    
    nitr = iterations
    
    zv_k_mean = 0.0
    zv_k_prec = inv(5.0)
    
    xv_k_mean = 0.0
    xv_k_prec = inv(5.0)
    
    zp_a = 10000000.0
    zp_b = inv(0.001)
    
    yp_a = 100.0
    yp_b = inv(0.1)
    
    marginals = Dict{Any, Any}(
        :xv_xv_min => ProbabilityDistribution(Multivariate, GaussianMeanVariance, m = [ 0.0, 0.0 ], v = [ 5.0 0.0; 0.0 5.0 ]),
        :zv => ProbabilityDistribution(Univariate, GaussianMeanPrecision, m = zv_k_mean, w = zv_k_prec),
        :xv => ProbabilityDistribution(Univariate, GaussianMeanPrecision, m = xv_k_mean, w = xv_k_prec),
        :zp => ProbabilityDistribution(Univariate, Gamma, a = zp_a, b = zp_b),
        :yp => ProbabilityDistribution(Univariate, Gamma, a = yp_a, b = yp_b)
    )
    
    fe = Array{Float64}(undef, length(observations), nitr)
    zm = []
    xm = []
    
    progress = ProgressMeter.Progress(length(observations))
    
    for (i, observation) in enumerate(observations)
        
        data = Dict(
            :zv_min_mean => zv_k_mean,
            :zv_min_prec => zv_k_prec,
            :xv_min_mean => xv_k_mean,
            :xv_min_prec => xv_k_prec,
            :zp_shape => zp_a,
            :zp_rate => zp_b,
            :yp_shape => yp_a,
            :yp_rate => yp_b,
            :yv => observation
        )
        
        for j in 1:nitr
            stepX!(data, marginals)
            stepZ!(data, marginals)
            stepZP!(data, marginals)
            stepYP!(data, marginals)
            fe[i, j] = freeEnergy(data, marginals)
        end
        
        push!(zm, marginals[:zv])
        push!(xm, marginals[:xv])
        
        zv_k_mean = ForneyLab.unsafeMean(marginals[:zv])
        zv_k_prec = inv(ForneyLab.unsafeVar(marginals[:zv]))
        xv_k_mean = ForneyLab.unsafeMean(marginals[:xv])
        xv_k_prec = inv(ForneyLab.unsafeVar(marginals[:xv]))

        zp_a = marginals[:zp].params[:a]
        zp_b = marginals[:zp].params[:b]
        yp_a = marginals[:yp].params[:a]
        yp_b = marginals[:yp].params[:b]
        if showprogress
            ProgressMeter.next!(progress)
        end
    end
    
    return (zm, xm, fe)
end


function extract_posteriors(T, results)
    zm, xm, fe = results

    @assert length(zm) === length(xm) === T

    emz = ForneyLab.unsafeMean.(zm)
    evz = ForneyLab.unsafeVar.(zm)

    emx = ForneyLab.unsafeMean.(xm)
    evx = ForneyLab.unsafeVar.(xm)

    ez = map((e) -> ReactiveMP.NormalMeanVariance(e...), zip(emz, evz))
    ex = map((e) -> ReactiveMP.NormalMeanVariance(e...), zip(emx, evx))

    return (z = ez, x = ex)
end

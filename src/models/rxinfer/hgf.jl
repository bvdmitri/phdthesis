# NOTE: this file is not included in the project by default, 
# it must be included explicitly in the notebook experiments

# We create a single-time step of corresponding state-space process to
# perform online learning (filtering)
@model function hgf(kappa, omega)
    
    # Priors from previous time step for `z_precision`
    zpprior = datavar(Float64, 2)
    zp ~ Gamma(zpprior[1], zpprior[2])
    
    # Priors from previos time step for `y_precision`
    ypprior = datavar(Float64, 2)
    yp ~ Gamma(ypprior[1], ypprior[2])
    
    # Priors from previous time step for `z`
    zt_minprior = datavar(Float64, 2)
    zt_min ~ Normal(mean = zt_minprior[1], precision = zt_minprior[2])
    
    # Priors from previous time step for `x`
    xt_minprior = datavar(Float64, 2)
    xt_min ~ Normal(mean = xt_minprior[1], precision = xt_minprior[2])

    # Higher layer is modelled as a random walk 
    zt ~ Normal(mean = zt_min, precision = zp)
    
    # Lower layer is modelled with the `GCV` node
    gcvnode, xt ~ GCV(xt_min, zt, kappa, omega)
    
    # Noisy observations 
    y = datavar(Float64)
    y ~ Normal(mean = xt, precision = yp)
    
    return gcvnode
    
end

@constraints function hgfconstraints() 
    q(yp, xt, zt, xt_min) = q(xt, xt_min)q(zt)q(yp)
    q(zt, zt_min, zp) = q(zt, zt_min)q(zp)
end

function run_inference(model, data; iterations = 5, free_energy = false)
    
    v_shape_scale(something) = SA[ shape(something), scale(something) ]
    v_mean_precision(something) = SA[ mean(something), precision(something) ]

    autoupdates = @autoupdates begin
        zpprior = v_shape_scale(q(zp))
        ypprior = v_shape_scale(q(yp))
        zt_minprior = v_mean_precision(q(zt))
        xt_minprior = v_mean_precision(q(xt))
    end

    return rxinference(
        model         = model,
        constraints   = hgfconstraints(),
        data          = (y = data, ),
        autoupdates   = autoupdates,
        keephistory   = length(data),
        historyvars    = (
            xt = KeepLast(),
            zt = KeepLast(),
        ),
        initmarginals = (
            zp = Gamma(10000000.0, 0.001),
            yp = Gamma(100.0, 0.1),
            zt = NormalMeanVariance(0.0, 5.0),
            xt = NormalMeanVariance(0.0, 5.0),
        ), 
        iterations    = iterations,
        free_energy   = free_energy,
        autostart     = true,
        callbacks     = (
            after_model_creation = (model, returnval) -> begin 
                gcvnode = returnval
                setmarginal!(gcvnode, :y_x, MvNormalMeanCovariance([ 0.0, 0.0 ], [ 5.0, 5.0 ]))
            end,
        )
    )
end

function extract_posteriors(T, results)
    @assert length(results.history[:zt]) === T
    @assert length(results.history[:xt]) === T
    return (
        z = results.history[:zt],
        x = results.history[:xt]
    )
end

function run_inference(model, datastream)
    
    v_shape_scale(posterior) = [ shape(posterior), scale(posterior) ]
    v_mean_precision(posterior) = [ mean(posterior), precision(posterior) ]

    autoupdates = @autoupdates begin
        prior_σₜ   = v_shape_scale(q(σₜ))
        prior_ωₜ   = v_shape_scale(q(ωₜ))
        prior_zₜ₋₁ = v_mean_precision(q(zₜ))
        prior_xₜ₋₁ = v_mean_precision(q(xₜ))
    end

    # Returns a reactive inference engine, that subscribes
    # to the `datastream` and performs continual inference
    # as soon as new measurements are available
    return rxinference(
        model         = model,
        constraints   = hgfconstraints(),
        datastream    = datastream,
        autoupdates   = autoupdates, 
        # ...
    )
end

# We create a single-time step of corresponding state-space process to
# perform online learning (filtering)
@model function hgf(κ, ζ)
    
    # Priors from previous time step for `σₜ`
    prior_σₜ = datavar(Float64, 2)
    σₜ ~ Gamma(prior_σₜ[1], prior_σₜ[2])
    
    # Priors from previos time step for `ωₜ`
    prior_ωₜ = datavar(Float64, 2)
    ωₜ ~ Gamma(prior_ωₜ[1], prior_ωₜ[2])
    
    # Priors from previous time step for `z`, which is a second layer
    prior_zₜ₋₁ = datavar(Float64, 2)
    zₜ₋₁ ~ Normal(mean = prior_zₜ₋₁[1], precision = prior_zₜ₋₁[2])

    # Priors from previous time step for `x`
    prior_xₜ₋₁ = datavar(Float64, 2)
    xₜ₋₁ ~ Normal(mean = prior_xₜ₋₁[1], precision = prior_xₜ₋₁[2])

    # The second layer is modeled as a random walk 
    zₜ ~ Normal(mean = zₜ₋₁, precision = σₜ)
    
    # The first layer is modeled with the `GCV` node
    xₜ ~ GCV(xₜ₋₁, zₜ, κ, ζ)
    
    # Noisy observations 
    y = datavar(Float64)
    y ~ Normal(mean = xₜ, precision = ωₜ)
end


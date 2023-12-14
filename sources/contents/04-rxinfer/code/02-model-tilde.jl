@model function model_name(; n, ...)
    ...
    # The `tilde` operator creates new random variables, if necessary
    # `μ` is modeled by the Normal distribution
    μ ~ Normal(0.0, 1.0)
    # `γ` is modeled by the Gamma distribution
    γ ~ Gamma(1.0, 1.0)

    # `y` are observations
    y = datavar(n)
    
    for i in 1:n
        # `y` are modelled by the Normal distribution
        # with the mean parameter `μ` and the precision `γ`
        y[i] ~ Normal(mean = μ, precision = γ)
    end
    ...
end

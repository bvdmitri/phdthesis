# A custom message update rule for the GCV node with the naive mean-field 
# factorization assumption 
@rule GCV(:y, Marginalisation) (q_x::Any, q_z::Any, q_κ::Any, q_ζ::Any) = begin
    x_mean        = mean(q_x)
    z_mean, z_var = mean_var(q_z)
    κ_mean, κ_var = mean_var(q_κ)
    ζ_mean, ζ_var = mean_var(q_ζ)

    ξ = κ_mean^2 * z_var + z_mean^2 * κ_var + z_var * κ_var
    A = exp(-ζ_mean + ζ_var / 2)
    B = exp(-κ_mean * z_mean + ξ / 2)

    # The resulting message is of type `Normal` 
    # with mean-variance parametrization
    return NormalMeanVariance(x_mean, inv(A * B))
end

# The model accepts the number of time steps `T`,
# the dimensionality of the state vector `dims`,
# state transition matrix `A`, observational matrix `B`,
# and the noise components `Σ` and `Ω`.
@model function linear_dynamical_system(T, dims, A, B, Σ, Ω)     
    # Create a sequence of random variables with length `T`
    s = randomvar(T) 

    # Create a sequence of observations with length `T`
    y = datavar(Vector{Float64}, T) 

    # Define a prior over the first state
    s[1] ~ MvNormal(mean = zeros(dims), covariance = 100I(dims)) 
    y[1] ~ B * s[1] + MvNormal(mean = zeros(dims), covariance = Ω)
    
    # Iterate over all remaining states
    for t in 2:T
        s[t] ~ A * s[t - 1] + MvNormal(mean = zeros(dims), covariance = Σ)
        y[t] ~ B * s[t] + MvNormal(mean = zeros(dims), covariance = Ω)    
    end
end


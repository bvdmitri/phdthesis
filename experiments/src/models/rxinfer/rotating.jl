# NOTE: this file is not included in the project by default, 
# it must be included explicitly in the notebook experiments

# Hot fixes for `ReactiveMP`
# These fixes are needed for the `StaticArray` library, this should be eventually 
# ported to the main code of the `ReactiveMP`, but I decided to keep it here for now
ReactiveMP.cholinv(x::UniformScaling) = inv(x.λ) * I

@model function rotating(T, environment)     
    # We create a vector of random variables with length `T`
    s = randomvar(T) 

    d = environment.d

    # Create a vector of observations with length `T`
    y = datavar(Vector{Float64}, T) 

    # We create a `constvar` references for constants in our model
    # to hint inference engine and to make it a little bit more efficient
    cA = constvar(environment.A)
    cB = constvar(environment.B)
    cP = constvar(environment.P)
    cQ = constvar(environment.Q)

    # Set a prior distribution for s[1]
    s[1] ~ MvNormal(mean = zeros(d), covariance = Matrix(100 * diageye(d))) 
    y[1] ~ MvNormal(mean = cB * s[1], covariance = cQ)
    
    # Iterate over all remaining states
    for t in 2:T
        s[t] ~ MvNormal(mean = cA * s[t - 1], covariance = cP)
        y[t] ~ MvNormal(mean = cB * s[t], covariance = cQ)    
    end
end

function extract_posteriors(T, results)
    return results.posteriors[:s]
end

# NOTE: this file is not included in the project by default, 
# it must be included explicitly in the notebook experiments

# Hot fixes for `ReactiveMP`
# These fixes are needed for the `StaticArray` library, this should be eventually 
# ported to the main code of the `ReactiveMP`, but I decided to keep it here for now
ReactiveMP.cholinv(x::UniformScaling) = inv(x.λ) * I
ReactiveMP.cholinv(x::SMatrix) = inv(x)
ReactiveMP.fastcholesky(x::SMatrix) = ReactiveMP.fastcholesky(Matrix(x))

@model function rotating(T, A, B, P, Q)     
    # We create a vector of random variables with length n
    s = randomvar(T) 

    # Create a vector of observations with length n
    y = datavar(Vector{Float64}, T) 

    # We create a `constvar` references for constants in our model
    # to hint inference engine and to make it a little bit more efficient
    cA = constvar(A)
    cB = constvar(B)
    cP = constvar(P)
    cQ = constvar(Q)

    d = size(A, 1)

    # Set a prior distribution for s[1]
    s[1] ~ MvGaussianMeanCovariance(zeros(4), Matrix(100I(d))) 
    y[1] ~ MvGaussianMeanCovariance(cB * s[1], cQ)

    for t in 2:T
        s[t] ~ MvGaussianMeanCovariance(cA * s[t - 1], cP)
        y[t] ~ MvGaussianMeanCovariance(cB * s[t], cQ)    
    end
end

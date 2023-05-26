# NOTE: this file is not included in the project by default, 
# it must be included explicitly in the notebook experiments

# Hot fixes for `ReactiveMP`
# These fixes are needed for the `StaticArray` library, this should be eventually 
# ported to the main code of the `ReactiveMP`, but I decided to keep it here for now
ReactiveMP.cholinv(x::UniformScaling) = inv(x.λ) * I
ReactiveMP.cholinv(x::SMatrix) = inv(x)
ReactiveMP.fastcholesky(x::SMatrix) = ReactiveMP.fastcholesky(Matrix(x))

# Slightly optimized version of the model used in the thesis
# - state + N(0, precision) is replaced with N(state, precision) as they are identical, but generate less nodes
# - arrays are replaced with their static array equivalents with the SA prefix
# - variable names are different, but that is minor IMO
@model function double_pendulum(T, s_start)
    
    s = randomvar(T)
    y = datavar(Float64, T)
    σ ~ Gamma(0.001, 100.0)
    
    Σ = constvar(SMatrix{4, 4}(1000 * diageye(4)))
    c = constvar(SA[ 0.0, 1.0, 0.0, 0.0 ])
    
    s[1] ~ MvNormal(mean = s_start, covariance = SMatrix{4, 4}(diageye(4)))
    y[1] ~ Normal(mean = dot(s[1], c), precision = σ)
    
    for t in 2:T
        s[t] ~ MvNormal(mean = f(s[t - 1]), precision = Σ)
        y[t] ~ Normal(mean = dot(s[t], c), precision = σ)
    end
end

@meta function double_pendulum_meta()
    f() -> Linearization()
end

@constraints function double_pendulum_constraints()
    q(s, σ) = q(s)q(σ)
end

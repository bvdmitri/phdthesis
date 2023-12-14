# The model accepts number of time-steps `T` and the initial state guess 
# for the first time step
@model function double_pendulum(T, s_start)

  s = randomvar(T)
  y = datavar(Float64, T)

  # The precision of the measurement noise is unknown
  # We use an uninformed prior with mean = 0.1 and variance = 10.0
  ω ~ Gamma(0.001, 100.0)

  # We assume relatively high state-transition precision
  # `zeros(4)` refers to a zeroed vector of length 4
  # `I(4)` refers to a diagonal matrix of size 4x4
  μσ = zeros(4)
  Σσ = 1000 * I(4)
  # We observational function `g` is the dot product with the vector `c`
  # That ensures that we observe only the `θ₂` component of the state
  c = [0.0, 1.0, 0.0, 0.0]

  # Define a prior over the first state 
  s[1] ~ MvNormal(mean = s_start, covariance = I(4))
  y[1] ~ dot(s[1], c) + Normal(mean = 0.0, precision = ω)

  for t in 2:T
    # The `f` function is defined globally and represents the discretized
    # differential equations of the motion of the system
    s[t] ~ f(s[t - 1]) + MvNormal(mean = μσ, precision = Σσ)
    y[t] ~ dot(s[t], c) + Normal(mean = 0, precision = ω)
  end
end

@constraints function double_pendulum_constraints()
    # The factorization constraint assumes that states `s` and the unknown 
    # noise component `ω` are jointly independent
    q(s, ω) = q(s)q(ω)
end

@meta function double_pendulum_meta()
    # We approximate the non-linear state-transition function `f`
    # with its linearized version locally
    f() -> Linearization()
end

results = inference(
    model       = double_pendulum(T, first_state_prior),
    data        = (y = observations, ),
    meta        = double_pendulum_meta(),
    constraints = double_pendulum_constraints(),
    # ...
)

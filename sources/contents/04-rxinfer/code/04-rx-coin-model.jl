# We use the `@model` macro to accept a regular Julia function as input.
# In this example, the model accepts two keyword argument 
# - `n` denotes the number of observations in the data set 
# - `priorθ` specifies a prior distribution over the `θ` parameter
@model function rx_coin_model()

  # We create `a` and `b` variables explicitly, such that 
  # we can continuously update the prior over the `θ` variable
  # as soon as we gather more and more data
  a = datavar(Float64)
  b = datavar(Float64)
  θ ~ Beta(a, b)

  # We have only a single observation `y` at each point
  y = datavar(Float64)
  y ~ Bernoulli(θ)

end

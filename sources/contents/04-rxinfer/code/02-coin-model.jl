# We use the `@model` macro to accept a regular Julia function as input.
# In this example, the model accepts two keyword argument 
# - `n` denotes the number of observations in the data set 
# - `priorθ` specifies a prior distribution over the `θ` parameter
@model function coin_model(; n, priorθ)

  # We use the tilde operator to define a probabilistic relationship between
  # random variables and data inputs. It automatically creates a new random 
  # variable in the current model and the corresponding factor nodes
  θ ~ priorθ

  # A sequence of observations of fixed length `n`
  y = datavar(Float64, n)

  # It is possible to use fixed control expressions, 
  # such as for loops or if statements
  for i in 1:n
    # Each observation is modeled by the `Bernoulli` distribution 
    # that is governed by the `θ` parameter
    y[i] ~ Bernoulli(θ)
  end

end

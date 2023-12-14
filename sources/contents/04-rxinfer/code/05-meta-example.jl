# Define a simple nonlinear function as an example
nonlinear_function(θ) = θ ^ 2

@model function nonlinear_coin_model(; n, priorθ)

  θ ~ priorθ

  y = datavar(Float64, n)

  for i in 1:n
    # We pass the `θ` parameter through a nonlinear deterministic function
    y[i] ~ Bernoulli(nonlinear_function(θ))
  end

end

# The `@meta` block defines computational strategy for certain factor nodes 
# in the factor graph representation of the probabilistic model
meta = @meta begin 
  # The `Conjugate-Computational Variational Inference` depends on
  # hyperparemeters indicated as `hyperparams...`
  nonlinear_function() -> CVI(hyperparams...)
end

dataset = readfile("dataset.txt")

result = inference(
  model = coin_model(priorθ = Uniform(0, 1), n = length(dataset)),
  data  = (y = dataset,)
  meta  = meta
)

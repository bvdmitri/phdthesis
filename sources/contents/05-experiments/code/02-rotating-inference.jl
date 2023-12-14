# The `inference` function in its simplistic form accepts a probabilistic `model` 
# together with `data` and computes variational posteriors for all hidden 
# states in the given probabilistic model
results = inference(
    model = linear_dynamical_system(T, dims, A, B, Σ, Ω), 
    data = (y = observations, )
)

constraints = @constraints begin
  # The posterior over `θ` variable should be in a form of the Beta distribution 
  q(θ) :: Beta
end

result = inference(
  model = coin_model(priorθ = Uniform(0, 1), n = length(dataset)),
  constraints = constraints,
  data  = (y = dataset,)
)

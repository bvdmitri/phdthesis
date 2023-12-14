# First we read (or create) our static dataset, which is an array of `0` and `1`s
dataset = readfile("dataset.txt")

# The `inference` function accepts several keyword arguments, some of which
# - `model` arguments expects a valid model specification from the `@model` macro
# - `data` argument expects a named tuple with static data 
#          for each `datavar` in the model
result = inference(
  model = coin_model(priorθ = Uniform(0, 1), n = length(dataset)),
  data  = (y = dataset,)
)

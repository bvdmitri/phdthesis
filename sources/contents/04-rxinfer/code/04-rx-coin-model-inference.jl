# The `datastream` must be in the form of an observable
# but the true nature of the data generation process is not that important
# It can be a stream from the internet or a signal from a sensor
datastream = readstream("http://coin-model.com")

# The `rxinference` function has a similar structure with the `inference` function
# but accepts only observables for the model observations
rxresults = rxinference(
  model = rx_coin_model(),
  datastream = datastream,
  autoupdates = autoupdates,
)

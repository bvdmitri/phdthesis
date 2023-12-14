# An example of the inference procedure with a fixed-length dataset 
# of observations of the double pendulum dynamical system
results = inference(
    model       = double_pendulum(T, first_state_prior),
    data        = (y = observations, ),
    meta        = double_pendulum_meta(),
    constraints = double_pendulum_constraints(),
    # ...
)

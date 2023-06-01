module RxInferThesisExperiments

export state_transition

"""
Generates an efficient version of the state transition discretization for various environments.
Returns a function of the state.
"""
function state_transition end

include("environments/doublependulum.jl")
include("environments/hgf.jl")

# The thesis uses the amse metric
include("metrics/amse.jl")

# Utility functions for benchmarks
include("benchmarks.jl")

end

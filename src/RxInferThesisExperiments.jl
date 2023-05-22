module RxInferThesisExperiments

export state_transition

"""
Generates an efficient version of the state transition discretization for various environments.
Returns a function of the state.
"""
function state_transition end

include("environments/doublependulum.jl")

end

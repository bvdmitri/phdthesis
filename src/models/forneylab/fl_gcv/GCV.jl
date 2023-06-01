module GCV

using ForneyLab
using HCubature
using FastGaussQuadrature
using ForwardDiff

include("approximations.jl")
include("gaussian_controlled_variance.jl")
include("rules_prototypes.jl")
include("update_rules.jl")

end  # module GCV

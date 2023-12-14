# The `@constraints` macro accepts a set of expression
# enclosed in the `begin ... end` block and each either 
# -a factorization constraint in the form `q(set..) = q(subset1..)q(subset2..)..`
# -a functional form constraint in the form `q(variable) :: FunctionalForm`
@constraints begin 
  # A factorization constraint for the joint 
  # variational distribution `q(x, y, z)`
  q(x, y, z) = q(x, y)q(z)
  # A functional form constraint for 
  # the variational distribution for the variable `x`
  q(x) :: Normal
end

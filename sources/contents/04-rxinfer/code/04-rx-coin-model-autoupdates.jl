autoupdates = @autoupdates begin
  # Both `a` and `b` variables in the model specification 
  # will be updated automatically with the parameters of the 
  # corresponding posterior `q(θ)` (as soon as new posterior is available)
  a, b = params(q(θ))
end

# The `@constraints` macro accepts Julia functions in order to support
# dynamic constraints construction
@constraints function create_constraints(is_mean_field, is_normal)

  # Optionally, assign the factorization constraints
  if is_mean_field 
    q(s) = q(s[begin])..q(s[end])
  end

  # Optionally, assign the functional form constraint
  if is_normal 
    q(s) :: Normal 
  end
end

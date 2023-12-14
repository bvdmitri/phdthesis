# Extra factorization constraints 
@constraints function double_pendulum_constraints()
    # The factorization constraint assumes that states `s` and the unknown 
    # noise component `ω` are jointly independent
    q(s, ω) = q(s)q(ω)
end

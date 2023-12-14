# Extra factorization constraints 
@constraints function hgfconstraints()
    # The factorization constraint assumes that states `z`, states `x`
    # and the unknown noise components `σₜ` and `ωₜ` are jointly independent
    q(ωₜ, xₜ₋₁, xₜ, zₜ) = q(ωₜ)q(xₜ₋₁, xₜ)q(zₜ)
    q(σₜ, zₜ₋₁, zₜ) = q(σₜ)q(zₜ₋₁, zₜ)
end


# Extra approximation methods 
@meta function double_pendulum_meta()
    # We approximate the non-linear state-transition function `f`
    # with its linearized version locally
    f() -> Linearization()
end

@model function model_name(...)
    ...
    # Creates a single data input called `y` of type `Float64`
    y = datavar(Float64)     

    # Creates a single data input called `y` of type `Vector{Float64}`
    y = datavar(Vector{Float64}) 

    # Returns a vector of `y_i` data inputs of length `n`
    y = datavar(Float64, n) 

    # Returns a matrix of `y_i_j` data inputs of size `(n, m)`
    y = datavar(Float64, n, m) 
    ...
end

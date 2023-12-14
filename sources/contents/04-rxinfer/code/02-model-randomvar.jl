@model function model_name(...)
    ...
    # Creates a single random variable called `s` 
    s = randomvar()     

    # Creates a single random variable called `s`
    s = randomvar() 

    # Returns a vector of `s_i` random variables of length `n`
    s = randomvar(n) 

    # Returns a matrix of `s_i_j` random variables of size `(n, m)`
    s = randomvar(n, m) 
    ...
end

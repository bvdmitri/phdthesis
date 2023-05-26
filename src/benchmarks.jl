# Utility functions for benchmarks across different packages

export compute_emse

emse_range(seed) = (seed):(seed + 10)

function compute_emse(f, seed)
    erange = emse_range(seed)
    return 1.0 / length(erange) * mapreduce(f, +, erange)
end

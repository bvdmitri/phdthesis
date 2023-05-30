# Utility functions for benchmarks across different packages

using DrWatson, DataFrames

export compute_emse, prepare_benchmarks_table

emse_range(seed) = (seed):(seed+10)

function compute_emse(f, seed)
    erange = emse_range(seed)
    return 1.0 / length(erange) * mapreduce(f, +, erange)
end


## Table analysis

to_ms_str(time) = string(round(time / 1_000_000, digits=4), "ms")

function benchmark_timings(key)
    return (data) -> begin
        benchmark = data[key]
        t_execution_min = minimum(benchmark).time
        t_execution_mean = mean(benchmark).time
        t_gc_min = minimum(benchmark).gctime
        return to_ms_str.((t_execution_min, t_execution_mean, t_gc_min))
    end
end

function prepare_benchmarks_table(folder)
    white_list = ["T", "seed", "niterations", "amse", "emse"]
    special_list = [
        :inference => benchmark_timings("benchmark_inference"),
        :creation => benchmark_timings("benchmark_modelcreation"),
    ]
    results = collect_results(folder; white_list=white_list, special_list=special_list)

    return select!(results, Not(:path))
end

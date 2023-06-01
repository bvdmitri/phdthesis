# Utility functions for benchmarks across different packages

using DrWatson, DataFrames

export compute_emse, prepare_benchmarks_table, to_ms_str

emse_range(seed) = (seed):(seed+10)

function compute_emse(f, seed)
    erange = emse_range(seed)
    return 1.0 / length(erange) * mapreduce(f, +, erange)
end


## Table analysis

to_ms_str(time; digits = 4) = string(round(time / 1_000_000, digits=digits), "ms")

function benchmark_timings_str(key)
    return (data) -> begin
        return to_ms_str.(benchmark_timings(key)(data))
    end
end

function benchmark_timings(key)
    return (data) -> begin
        benchmark = data[key]
        t_execution_min = minimum(benchmark).time
        t_execution_mean = mean(benchmark).time
        t_gc_min = minimum(benchmark).gctime
        return (t_execution_min, t_execution_mean, t_gc_min)
    end
end


function prepare_benchmarks_table(folder)
    # white_list = ["T", "seed", "niterations", "amse", "emse"]
    black_list = [ "states", "e_states", "observations", "benchmark_inference", "benchmark_modelcreation" ]
    special_list = [
        :inference => benchmark_timings("benchmark_inference"),
        :creation => benchmark_timings("benchmark_modelcreation"),
    ]
    results = collect_results(folder; black_list=black_list, special_list=special_list)

    return select!(results, Not(:path))
end

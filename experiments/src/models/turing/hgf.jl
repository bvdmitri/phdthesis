# NOTE: this file is not included in the project by default, 
# it must be included explicitly in the notebook experiments

using Logging

import ReactiveMP

Turing.setprogress!(false)

@model function HGF(observation, zt_min_prior, xt_min_prior, z_std_prior, y_std_prior, kappa, omega)
    # Priors
    z_std ~ z_std_prior
    y_std ~ y_std_prior

    zt_min ~ zt_min_prior
    xt_min ~ xt_min_prior

    zt ~ Normal(zt_min, sqrt(inv(z_std)))
    xt ~ Normal(xt_min, sqrt(exp(kappa * zt + omega)))

    observation ~ Normal(xt, sqrt(inv(y_std)))
end

function extract_params_for_next_step(rng, model, chain::Turing.Chains)
    sumstats = Turing.summarize(chain, Turing.mean, Turing.std)
    
    xt_index = findnext(e -> e === :xt, sumstats.nt.parameters, 1)
    zt_index = findnext(e -> e === :zt, sumstats.nt.parameters, 1)
    z_std_index = findnext(e -> e === :z_std, sumstats.nt.parameters, 1)
    y_std_index = findnext(e -> e === :y_std, sumstats.nt.parameters, 1)
    
    z_std_mean = sumstats.nt.mean[z_std_index]
    z_std_var = max(abs2(sumstats.nt.std[z_std_index]), 0.1) # Otherwise is very unstable
    y_std_mean = sumstats.nt.mean[y_std_index]
    y_std_var = max(abs2(sumstats.nt.std[y_std_index]), 0.1) # Otherwise is very unstable
    
    return (
        xt_min_prior = Normal(sumstats.nt.mean[xt_index], sumstats.nt.std[xt_index]),
        zt_min_prior = Normal(sumstats.nt.mean[zt_index], sumstats.nt.std[zt_index]),
        z_std_prior  = Gamma(z_std_mean ^ 2 / z_std_var, z_std_var / z_std_mean),
        y_std_prior  = Gamma(y_std_mean ^ 2 / y_std_var, y_std_var / y_std_mean),
    )
end

function extract_params_for_next_step(rng, model, result::Turing.MultivariateTransformed)
    _, sym2range = bijector(model, Val(true));
    dist = result.dist
    
    xt_index = sym2range[:xt][1]
    zt_index = sym2range[:zt][1]
    z_std_index = sym2range[:z_std][1]
    y_std_index = sym2range[:y_std][1]
    
    ms = mean(dist)
    vs = diag(cov(dist))
        
    @assert length(xt_index) === 1
    @assert length(zt_index) === 1
    @assert length(z_std_index) === 1
    @assert length(y_std_index) === 1
    
    z_std_q = transformed(Normal(ms[z_std_index][1], vs[z_std_index][1]), result.transform.bs[1])
    z_std_prior = fit(Gamma, rand(rng, z_std_q, 20_000))
        
    y_std_q = transformed(Normal(ms[y_std_index][1], vs[y_std_index][1]), result.transform.bs[2])
    y_std_prior = fit(Gamma, rand(rng, y_std_q, 20_000))

    return (
        xt_min_prior = Normal(ms[xt_index][1], sqrt(vs[xt_index][1])),
        zt_min_prior = Normal(ms[zt_index][1], sqrt(vs[zt_index][1])),
        z_std_prior  = z_std_prior,
        y_std_prior  = y_std_prior
    )
    
end

function execute_inference(method::NUTS, model; rng, nsamples)
    return sample(rng, model, method, nsamples);
end

function execute_inference(method::ADVI, model; rng, nsamples)
    return vi(model, method)
end

function run_inference(modelf, data; nsamples = 1000, method = NUTS(), rng = StableRNG(42))
    # Disable turing's logger
    return with_logger(NullLogger()) do
        zt_min_prior = Normal(0.0, sqrt(5.0))
        xt_min_prior = Normal(0.0, sqrt(5.0))
        z_std_prior = Gamma(10000000.0, 0.001)
        y_std_prior = Gamma(100.0, 0.1)

        results = []

        @showprogress for observation in data
    
            model = modelf(observation, zt_min_prior, xt_min_prior, z_std_prior, y_std_prior)
            result = execute_inference(method, model; rng = rng, nsamples = nsamples)
            stats = extract_params_for_next_step(rng, model, result)

            zt_min_prior = stats[:zt_min_prior]
            xt_min_prior = stats[:xt_min_prior]
            z_std_prior = stats[:z_std_prior]
            y_std_prior = stats[:y_std_prior]

            push!(results, stats)
        end

        return results
    end
end

function extract_posteriors(T, results)
    ez = getindex.(results, :zt_min_prior)
    ex = getindex.(results, :zt_min_prior)

    ecz = map(q -> ReactiveMP.NormalMeanVariance(mean(q), var(q)), ez)
    ecx = map(q -> ReactiveMP.NormalMeanVariance(mean(q), var(q)), ex)

    return (z = ecz, x = ecx)
end


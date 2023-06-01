import ForneyLab: SoftFactor, @ensureVariables, @symmetrical, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov, Univariate, Gaussian, prod!

export prod!, quadrature, NewtonMethod


function quadrature(g::Function,d::ProbabilityDistribution{Univariate,GaussianMeanVariance},p::Int64)
    sigma_points, sigma_weights = gausshermite(p)
    m, v = ForneyLab.unsafeMeanCov(d)
    result = 0.0
    for i=1:p
        result += sigma_weights[i]*g(m+sqrt(2*v)*sigma_points[i])/sqrt(pi)
    end
    return result
end

function NewtonMethod(g::Function, x_0::Array{Float64})

    grad_g = (x) -> ForwardDiff.gradient(g, x)
    mode   = gradientOptimization(g, grad_g, x_0, 0.01)

    cov  = cholinv(-ForwardDiff.hessian(g, mode))

    return mode, cov
end

function gradientOptimization(log_joint::Function, d_log_joint::Function, m_initial, step_size)
    dim_tot = length(m_initial)
    m_total = zeros(dim_tot)
    m_average = zeros(dim_tot)
    m_new = zeros(dim_tot)
    m_old = m_initial
    satisfied = false
    step_count = 0

    while !satisfied
        m_new = m_old .+ step_size.*d_log_joint(m_old)
        if log_joint(m_new) > log_joint(m_old)
            proposal_step_size = 10*step_size
            m_proposal = m_old .+ proposal_step_size.*d_log_joint(m_old)
            if log_joint(m_proposal) > log_joint(m_new)
                m_new = m_proposal
                step_size = proposal_step_size
            end
        else
            step_size = 0.1*step_size
            m_new = m_old .+ step_size.*d_log_joint(m_old)
        end
        step_count += 1
        m_total .+= m_old
        m_average = m_total ./ step_count
        if step_count > 10
            if sum(sqrt.(((m_new.-m_average)./m_average).^2)) < dim_tot*0.0001
                satisfied = true
            end
        end
        if step_count > dim_tot*250
            satisfied = true
        end
        m_old = m_new
    end

    return m_new
end

@symmetrical function prod!(x::ProbabilityDistribution{Univariate, Function},
               y::ProbabilityDistribution{Univariate, F1},
               z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F1<:Gaussian

    p = 20
    dist_y = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance},y)
    g(k) = exp(x.params[:log_pdf](k))
    normalization_constant = quadrature(g,dist_y,p)
    t(k) = k*g(k)/normalization_constant
    mean = quadrature(t,dist_y,p)
    s(k) = (k-mean)^2*g(k)/normalization_constant
    var = quadrature(s,dist_y,p)

    z.params[:m] = mean
    z.params[:v] = var
    
    # error(1)

    return z
end

# @symmetrical function prod!(x::ProbabilityDistribution{Univariate, Function},
#                y::ProbabilityDistribution{Univariate, Function},
#                z::ProbabilityDistribution{Univariate, GaussianMeanVariance}=ProbabilityDistribution(Univariate, GaussianMeanVariance, m=0.0,v=1.0)) where F1<:Gaussian
#
#     p = 20
#
#     g(k) = x.params[:log_pdf](k)+y.params[:log_pdf](k)
#     mean, var = NewtonMethod(g,0.0)
#
#     z.params[:m] = mean
#     z.params[:v] = var
#
#     return z
# end

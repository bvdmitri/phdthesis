import ForneyLab: SoftFactor, @ensureVariables, generateId, addNode!, associate!,
                  averageEnergy, Interface, Variable, slug, ProbabilityDistribution,
                  differentialEntropy, unsafeLogMean, unsafeMean, unsafeCov, unsafePrecision, unsafeMeanCov

export GaussianControlledVariance, averageEnergy, slug

"""
Description:
    A gaussian node where variance is controlled by a state that is passed
    through an exponential non-linearity.
    f(out,x,z,κ,ω) = N(out|x,exp(κz+ω))
Interfaces:
    1. out
    2. x (mean)
    3. z (state controlling the variance)
    4. κ
    5. ω
Construction:
    GaussianControlledVariance(out,x,z,κ,ω)
"""
mutable struct GaussianControlledVariance <: SoftFactor
    id::Symbol
    interfaces::Vector{Interface}
    i::Dict{Symbol,Interface}

    function GaussianControlledVariance(out, x, z, κ, ω; id=generateId(GaussianControlledVariance))
        @ensureVariables(out, x, z, κ, ω)
        self = new(id, Array{Interface}(undef, 5), Dict{Symbol,Interface}())
        addNode!(currentGraph(), self)
        self.i[:out] = self.interfaces[1] = associate!(Interface(self), out)
        self.i[:x] = self.interfaces[2] = associate!(Interface(self), x)
        self.i[:z] = self.interfaces[3] = associate!(Interface(self), z)
        self.i[:κ] = self.interfaces[4] = associate!(Interface(self), κ)
        self.i[:ω] = self.interfaces[5] = associate!(Interface(self), ω)

        return self
    end
end

slug(::Type{GaussianControlledVariance}) = "GCV"


# Average energy functional structured
function averageEnergy(::Type{GaussianControlledVariance}, marg_out_x::ProbabilityDistribution{Multivariate}, marg_z::ProbabilityDistribution{Univariate}, marg_κ::ProbabilityDistribution{Univariate}, marg_ω::ProbabilityDistribution{Univariate})
    m_out_x, cov_out_x = unsafeMeanCov(marg_out_x)
    m_z, var_z = unsafeMeanCov(marg_z)
    m_κ, var_κ = unsafeMeanCov(marg_κ)
    m_ω, var_ω = unsafeMeanCov(marg_ω)

    ksi = (m_κ^2)*var_z + (m_z^2)*var_κ + var_κ*var_z
    psi = (m_out_x[2]-m_out_x[1])^2 + cov_out_x[1,1]+cov_out_x[2,2]-cov_out_x[1,2]-cov_out_x[2,1]
    A = exp(-m_ω + var_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    0.5log(2*pi) + 0.5*(m_z*m_κ+m_ω) + 0.5*(psi*A*B)
end

#Average energy functional mean field
function averageEnergy(::Type{GaussianControlledVariance}, marg_out::ProbabilityDistribution{Univariate},marg_x::ProbabilityDistribution{Univariate}, marg_z::ProbabilityDistribution{Univariate}, marg_κ::ProbabilityDistribution{Univariate}, marg_ω::ProbabilityDistribution{Univariate})
    m_out, var_out = unsafeMeanCov(marg_out)
    m_x, var_x = unsafeMeanCov(marg_x)
    m_z, var_z = unsafeMeanCov(marg_z)
    m_κ, var_κ = unsafeMeanCov(marg_κ)
    m_ω, var_ω = unsafeMeanCov(marg_ω)

    ksi = (m_κ^2)*var_z + (m_z^2)*var_κ + var_κ*var_z
    psi = (m_out-m_x)^2 + var_x + var_out
    A = exp(-m_ω + var_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    0.5log(2*pi) + 0.5*(m_z*m_κ+m_ω) + 0.5*(psi*A*B)
end

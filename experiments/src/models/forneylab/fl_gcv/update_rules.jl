import ForneyLab: collectStructuredVariationalNodeInbounds, ultimatePartner,
                  currentInferenceAlgorithm, posteriorFactor,localEdgeToRegion, ultimatePartner, assembleClamp!,isApplicable,matches,outboundType, isClamped
export
  ruleSVBGaussianControlledVarianceOutNGDDD,
  ruleSVBGaussianControlledVarianceXGNDDD,
  ruleSVBGaussianControlledVarianceZDNDD,
  ruleSVBGaussianControlledVarianceΚDDND,
  ruleSVBGaussianControlledVarianceΩDDDN,
  ruleMGaussianControlledVarianceGGDDD,
  ruleSVBGaussianControlledVarianceOutNGDD,
  ruleSVBGaussianControlledVarianceXGNDD,
  ruleSVBGaussianControlledVarianceZDGGD,
  ruleSVBGaussianControlledVarianceΚDGGD,
  ruleSVBGaussianControlledVarianceΩDDN,
  ruleSVBGaussianControlledVarianceOutNEDDD,
  ruleSVBGaussianControlledVarianceMENDDD,
  ruleMGaussianControlledVarianceGGDD,
  ruleMGaussianControlledVarianceDGGD,
  ruleMGaussianControlledVarianceEGDDD,
  ruleMGaussianControlledVarianceGEDDD,
  ruleSVBGaussianMeanPrecisionMEND,
  ruleSVBGaussianMeanPrecisionOutNED,
  ruleMGaussianMeanPrecisionGED,
  ruleMGaussianMeanPrecisionEGD,
  ruleSPEqualityGaussianGCV,
  isApplicable,
  outboundType

function ruleSVBGaussianControlledVarianceOutNGDDD(dist_out::Nothing,
                                                   msg_x::Message{F, Univariate},
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_x.dist)
    m_x = dist_x.params[:m]
    v_x = dist_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)


    return Message(Univariate, GaussianMeanVariance, m=m_x, v=v_x+inv(A*B))
end

function ruleSVBGaussianControlledVarianceXGNDDD(msg_out::Message{F, Univariate},
                                                   dist_x::Nothing,
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_out = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},msg_out.dist)
    m_out = dist_out.params[:m]
    v_out = dist_out.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)

    return Message(Univariate, GaussianMeanVariance, m=m_out, v=v_out+inv(A*B))
end


function ruleSVBGaussianControlledVarianceZDNDD(dist_out_x::ProbabilityDistribution{Multivariate, F},
                                                dist_z::Nothing,
                                                dist_κ::ProbabilityDistribution{Univariate},
                                                dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)
    l_pdf(z) = -0.5*(m_κ*z + Psi*A*exp(-m_κ*z+0.5*v_κ*z^2/2))
    return Message(Univariate, Function, log_pdf=l_pdf)
end


function ruleSVBGaussianControlledVarianceΚDDND(dist_out_x::ProbabilityDistribution{Multivariate, F},
                                                dist_z::ProbabilityDistribution{Univariate},
                                                dist_κ::Nothing,
                                                dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)

    l_pdf(κ) = -0.5*(m_z*κ + Psi*A*exp(-m_z*κ+0.5*v_z*κ^2/2))
    return Message(Univariate, Function, log_pdf = l_pdf)
end


function ruleSVBGaussianControlledVarianceΩDDDN(dist_out_x::ProbabilityDistribution{Multivariate, F},
                                                dist_z::ProbabilityDistribution{Univariate},
                                                dist_κ::ProbabilityDistribution{Univariate},
                                                dist_ω::Nothing) where F<:Gaussian

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    B = exp(-m_κ*m_z + ksi/2)
    l_pdf(ω) = -0.5*(ω + Psi*B*exp(-ω))
    return Message(Univariate, Function, log_pdf = l_pdf)
end


function ruleMGaussianControlledVarianceGGDDD(msg_out::Message{F1, Univariate},
                                              msg_x::Message{F2, Univariate},
                                              dist_z::ProbabilityDistribution{Univariate},
                                              dist_κ::ProbabilityDistribution{Univariate},
                                              dist_ω::ProbabilityDistribution{Univariate}) where {F1 <: Gaussian, F2 <: Gaussian}
    dist_out = convert(ProbabilityDistribution{Univariate,GaussianMeanPrecision},msg_out.dist)
    dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanPrecision},msg_x.dist)
    m_x = dist_x.params[:m]
    w_x = dist_x.params[:w]
    m_out = dist_out.params[:m]
    w_out = dist_out.params[:w]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)
    W = [w_out+A*B -A*B; -A*B w_x+A*B]
    m = cholinv(W) * [m_out*w_out; m_x*w_x]

    return ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=m, w=W)

end

function ruleSVBGaussianControlledVarianceZDGGD(dist_out_x::ProbabilityDistribution{Multivariate,F1},
                                                msg_z::Message{F2,Univariate},
                                                msg_κ::Message{F3,Univariate},
                                                dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian, F2<:Gaussian,F3<:Gaussian}

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_κ, v_κ = unsafeMeanCov(msg_κ.dist)
    m_z, v_z = unsafeMeanCov(msg_z.dist)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)
    h(x) = -0.5*((x[1]-m_κ)^2/v_κ +(x[2]-m_z)^2/v_z + x[1]*x[2] + A*Psi*exp(-x[1]*x[2]))
    newton_m, newton_v = NewtonMethod(h,[m_κ; m_z])
    mean = newton_m[2] + newton_v[1,2]*inv(newton_v[1,1])*(m_κ-newton_m[1])
    var = newton_v[2,2] - newton_v[1,2]*inv(newton_v[1,1])*newton_v[1,2] + (newton_v[1,2]*inv(newton_v[1,1]))^2*v_κ


    return Message(Univariate, GaussianMeanVariance, m=mean ,v=var)
end

function ruleSVBGaussianControlledVarianceΚDGGD(dist_out_x::ProbabilityDistribution{Multivariate,F1},
                                                msg_z::Message{F2,Univariate},
                                                msg_κ::Message{F3,Univariate},
                                                dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian, F2<:Gaussian,F3<:Gaussian}

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_κ, v_κ = unsafeMeanCov(msg_κ.dist)
    m_z, v_z = unsafeMeanCov(msg_z.dist)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)
    h(x) = -0.5*((x[1]-m_κ)^2/v_κ +(x[2]-m_z)^2/v_z + x[1]*x[2] + A*Psi*exp(-x[1]*x[2]))
    newton_m, newton_v = NewtonMethod(h,[m_κ; m_z])
    mean = newton_m[1] + newton_v[1,2]*inv(newton_v[2,2])*(m_κ-newton_m[2])
    var = newton_v[1,1] - newton_v[1,2]*inv(newton_v[2,2])*newton_v[1,2] + (newton_v[1,2]*inv(newton_v[2,2]))^2*v_z


    return Message(Univariate, GaussianMeanVariance, m=mean ,v=var)
end

function ruleSVBGaussianControlledVarianceOutNEDDD(msg_out::Message{F, Univariate},
                                                   msg_x::Message{Function},
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    msg_x_prime = ruleSVBGaussianControlledVarianceOutNGDDD(nothing,msg_out,dist_z,dist_κ,dist_ω)
    approx_dist = msg_x_prime.dist*msg_x.dist.params[:log_pdf]
    approx_msg = Message(GaussianMeanVariance,m=approx_dist.params[:m],v=approx_dist.params[:v])
    return ruleSVBGaussianControlledVarianceOutNGDDD(nothing, approx_msg,dist_z,dist_κ,dist_ω)
end

function ruleSVBGaussianControlledVarianceMENDDD(msg_out::Message{Function},
                                                   msg_x::Message{F, Univariate},
                                                   dist_z::ProbabilityDistribution{Univariate},
                                                   dist_κ::ProbabilityDistribution{Univariate},
                                                   dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian

    msg_out_prime = ruleSVBGaussianControlledVarianceOutNGDDD(nothing,msg_x,dist_z,dist_κ,dist_ω)
    approx_dist = msg_out_prime.dist*msg_out.dist.params[:log_pdf]
    approx_msg = Message(GaussianMeanVariance,m=approx_dist.params[:m],v=approx_dist.params[:v])
    return ruleSVBGaussianControlledVarianceOutNGDDD(nothing, approx_msg,dist_z,dist_κ,dist_ω)
end

function ruleMGaussianControlledVarianceDGGD(dist_out_x::ProbabilityDistribution{Multivariate, F1},
                                             msg_z::Message{F2, Univariate},
                                             msg_κ::Message{F3, Univariate},
                                             dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian,F2<:Gaussian,F3<:Gaussian}

    dist_out_x = convert(ProbabilityDistribution{Multivariate,GaussianMeanVariance},dist_out_x)
    m = dist_out_x.params[:m]
    v = dist_out_x.params[:v]
    m_z, v_z = unsafeMeanCov(msg_z.dist)
    m_κ, v_κ = unsafeMeanCov(msg_κ.dist)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m[1]-m[2])^2+v[1,1]+v[2,2]-v[1,2]-v[2,1]
    A = exp(-m_ω+v_ω/2)
    h(x) = -0.5*((x[1]-m_κ)^2/v_κ +(x[2]-m_z)^2/v_z + x[1]*x[2] + A*Psi*exp(-x[1]*x[2]))
    newton_m, newton_v = NewtonMethod(h,[m_κ; m_z])

    return ProbabilityDistribution(Multivariate,GaussianMeanVariance,m=newton_m,v=newton_v)
end


function collectStructuredVariationalNodeInbounds(::GaussianControlledVariance, entry::ScheduleEntry)
    current_inference_algorithm = currentInferenceAlgorithm()
    interface_to_schedule_entry = current_inference_algorithm.interface_to_schedule_entry
    target_to_marginal_entry = current_inference_algorithm.target_to_marginal_entry

    inbounds = Any[]
    entry_posterior_factor = posteriorFactor(entry.interface.edge)
    local_edge_to_region = localEdgeToRegion(entry.interface.node)

    encountered_posterior_factors = Union{PosteriorFactor, Edge}[] # Keep track of encountered posterior factors
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        current_posterior_factor = posteriorFactor(node_interface.edge)

        if node_interface === entry.interface
            if entry.message_update_rule in [SVBGaussianControlledVarianceΚDGGD,SVBGaussianControlledVarianceZDGGD,SVBGaussianControlledVarianceOutNEDDD,SVBGaussianControlledVarianceMENDDD]
                push!(inbounds, interface_to_schedule_entry[inbound_interface])
            else
                push!(inbounds, nothing)
            end
        elseif isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        elseif current_posterior_factor === entry_posterior_factor
            # Collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        elseif !(current_posterior_factor in encountered_posterior_factors)
            # Collect marginal from marginal dictionary (if marginal is not already accepted)
            target = local_edge_to_region[node_interface.edge]
            push!(inbounds, target_to_marginal_entry[target])
        end

        push!(encountered_posterior_factors, current_posterior_factor)
    end

    return inbounds
end

function collectStructuredVariationalNodeInbounds(::GaussianMeanPrecision, entry::ScheduleEntry)
    current_inference_algorithm = currentInferenceAlgorithm()
    interface_to_schedule_entry = current_inference_algorithm.interface_to_schedule_entry
    target_to_marginal_entry = current_inference_algorithm.target_to_marginal_entry

    inbounds = Any[]
    entry_posterior_factor = posteriorFactor(entry.interface.edge)
    local_edge_to_region = localEdgeToRegion(entry.interface.node)

    encountered_posterior_factors = Union{PosteriorFactor, Edge}[] # Keep track of encountered posterior factors
    for node_interface in entry.interface.node.interfaces
        inbound_interface = ultimatePartner(node_interface)
        current_posterior_factor = posteriorFactor(node_interface.edge)

        if node_interface === entry.interface
            if (entry.message_update_rule == SVBGaussianMeanPrecisionOutNED) || (entry.message_update_rule == SVBGaussianMeanPrecisionMEND)
                push!(inbounds, interface_to_schedule_entry[inbound_interface])
            else
                # Ignore marginal of outbound edge
                push!(inbounds, nothing)
            end
        elseif isClamped(inbound_interface)
            # Hard-code marginal of constant node in schedule
            push!(inbounds, assembleClamp!(inbound_interface.node, ProbabilityDistribution))
        elseif current_posterior_factor === entry_posterior_factor
            # Collect message from previous result
            push!(inbounds, interface_to_schedule_entry[inbound_interface])
        elseif !(current_posterior_factor in encountered_posterior_factors)
            # Collect marginal from marginal dictionary (if marginal is not already accepted)
            target = local_edge_to_region[node_interface.edge]
            push!(inbounds, target_to_marginal_entry[target])
        end

        push!(encountered_posterior_factors, current_posterior_factor)
    end

    return inbounds
end

# Updates for equality
ruleSPEqualityGaussianGCV(msg_1::Message{F1},msg_2::Message{F2},msg_3::Nothing) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_1.dist,msg_2.dist))
ruleSPEqualityGaussianGCV(msg_1::Message{F2},msg_2::Message{F1},msg_3::Nothing) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_1.dist,msg_2.dist))
ruleSPEqualityGaussianGCV(msg_1::Message{F1},msg_2::Nothing,msg_3::Message{F2}) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_1.dist,msg_3.dist))
ruleSPEqualityGaussianGCV(msg_1::Message{F2},msg_2::Nothing,msg_3::Message{F1}) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_1.dist,msg_3.dist))
ruleSPEqualityGaussianGCV(msg_1::Nothing,msg_2::Message{F1},msg_3::Message{F2}) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_2.dist,msg_3.dist))
ruleSPEqualityGaussianGCV(msg_1::Nothing,msg_2::Message{F2},msg_3::Message{F1}) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_2.dist,msg_3.dist))

mutable struct SPEqualityGaussianGCV <: SumProductRule{Equality} end
outboundType(::Type{SPEqualityGaussianGCV}) = Message{Gaussian}
function isApplicable(::Type{SPEqualityGaussianGCV}, input_types::Vector{Type})
    nothing_inputs = 0
    gaussian_inputs = 0
    function_inputs = 0
    for input_type in input_types
        if input_type == Nothing
            nothing_inputs += 1
        elseif matches(input_type, Message{Gaussian})
            gaussian_inputs += 1
        elseif matches(input_type, Message{Function})
            function_inputs += 1
        end
    end

    return (nothing_inputs == 1) && (gaussian_inputs == 1) && (function_inputs == 1)
end

function isApplicable(::Type{SPEqualityGaussianGCV}, input_types::Vector{Type})
    nothing_inputs = 0
    function_inputs = 0
    for input_type in input_types
        if input_type == Nothing
            nothing_inputs += 1
        elseif matches(input_type, Message{Function})
            function_inputs += 1
        end
    end

    return (nothing_inputs == 1)  && (function_inputs == 2)
end


# # GaussianMeanPrecision updates
# function ruleSVBGaussianMeanPrecisionOutEND(msg_out::Message{ExponentialLinearQuadratic},
#                                             msg_mean::Message{F, Univariate},
#                                             dist_prec::ProbabilityDistribution) where F<:Gaussian
#
#     dist_out = msg_out.dist
#     message_prior = ruleSVBGaussianMeanPrecisionOutVGD(nothing, msg_mean,dist_prec)
#     dist_prior = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance},message_prior.dist)
#     approx_dist = dist_prior*msg_out.dist
#
#     return Message(GaussianMeanVariance, m=unsafeMean(approx_dist), v=unsafeCov(approx_dist) + cholinv(unsafeMean(dist_prec)))
#
# end

function ruleSVBGaussianMeanPrecisionOutNED(msg_out::Message{F,Univariate},
                                   msg_mean::Message{Function,Univariate},
                                   dist_prec::ProbabilityDistribution) where F<:Gaussian
    dist_mean = msg_mean.dist.params[:log_pdf]
    message_prior = ruleSVBGaussianPrecisionOutVGD(nothing, msg_out,dist_prec)
    dist_prior = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance},message_prior.dist)
    approx_dist = dist_prior*msg_mean.dist

    return Message(GaussianMeanVariance, m=unsafeMean(approx_dist), v=unsafeCov(approx_dist))
end

function ruleSVBGaussianMeanPrecisionMEND(msg_out::Message{Function, Univariate},
                                   msg_mean::Message{F, Univariate},
                                   dist_prec::ProbabilityDistribution) where F<:Gaussian

    dist_out = msg_out.dist.params[:log_pdf]
    message_prior = ruleSVBGaussianPrecisionOutVGD(nothing, msg_mean,dist_prec)
    dist_prior = convert(ProbabilityDistribution{Univariate, GaussianMeanVariance},message_prior.dist)
    approx_dist = dist_prior*msg_out.dist

    return Message(GaussianMeanVariance, m=unsafeMean(approx_dist), v=unsafeCov(approx_dist))
end


function ruleMGaussianMeanPrecisionGED(
    msg_out::Message{F, Univariate},
    msg_mean::Message{Function, Univariate},
    dist_prec::ProbabilityDistribution) where F<:Gaussian

    W_bar = unsafeMean(dist_prec)
    h(z) = msg_mean.dist.params[:log_pdf](z[2]) + logPdf(msg_out.dist,z[1])
    m_out = unsafeMean(msg_out.dist)
    z_0 = [m_out+sqrt.(1/W_bar)*randn(); m_out]
    m,V = NewtonMethod(h,z_0)


    return ProbabilityDistribution(Multivariate, GaussianMeanVariance, m=m, v=V)
end

function ruleMGaussianMeanPrecisionEGD(
    msg_out::Message{Function},
    msg_mean::Message{F, Univariate},
    dist_prec::ProbabilityDistribution) where F<:Gaussian

#     W_bar = unsafeMean(dist_prec)
#     h(z) = msg_out.dist.params[:log_pdf](z[1]) + logPdf(msg_mean.dist,z[2])
#     m_mean = unsafeMean(msg_mean.dist)
#     z_0 = [m_mean+sqrt.(1/W_bar)*randn(); m_mean]
#     m,V = NewtonMethod(h,z_0)
    
    p = 20
    d = ProbabilityDistribution(Univariate, GaussianMeanVariance, m = 0.0, v = 1.0)
    
    g = (z) -> exp(msg_out.dist.params[:log_pdf](z)) * exp(0.5 * z^2)

    normalization_constant = quadrature(g, d, p)
    t = (z) -> z * g(z) / normalization_constant
    mean = quadrature(t, d, p)
    s = (z) -> (z - mean)^2 * g(z) / normalization_constant
    var = quadrature(s, d, p)

    m_out = Message(GaussianMeanVariance, m=mean, v=var)

    return ruleMGaussianPrecisionGGD(m_out, msg_mean, dist_prec)
#     return ProbabilityDistribution(Multivariate, GaussianMeanVariance,m=m,v=V)
end

mutable struct SPEqualityGaussianGCV <: SumProductRule{Equality} end
outboundType(::Type{SPEqualityGaussianGCV}) = Message{GaussianMeanVariance}
function isApplicable(::Type{SPEqualityGaussianGCV}, input_types::Vector{Type})
  nothing_inputs = 0
  gaussian_inputs = 0
  function_inputs = 0
  for input_type in input_types
      if input_type == Nothing
          nothing_inputs += 1
      elseif matches(input_type, Message{Gaussian})
          gaussian_inputs += 1
      elseif matches(input_type, Message{Function})
          function_inputs += 1
      end
  end

  return (nothing_inputs == 1) && (gaussian_inputs == 1) && (function_inputs == 1)
end

ruleSPEqualityGaussianGCV(msg_1::Message{F1},msg_2::Message{F2},msg_3::Nothing) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_1.dist,msg_2.dist))
ruleSPEqualityGaussianGCV(msg_1::Message{F2},msg_2::Message{F1},msg_3::Nothing) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_1.dist,msg_2.dist))
ruleSPEqualityGaussianGCV(msg_1::Message{F1},msg_2::Nothing,msg_3::Message{F2}) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_1.dist,msg_3.dist))
ruleSPEqualityGaussianGCV(msg_1::Message{F2},msg_2::Nothing,msg_3::Message{F1}) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_1.dist,msg_3.dist))
ruleSPEqualityGaussianGCV(msg_1::Nothing,msg_2::Message{F1},msg_3::Message{F2}) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_2.dist,msg_3.dist))
ruleSPEqualityGaussianGCV(msg_1::Nothing,msg_2::Message{F2},msg_3::Message{F1}) where {F1<:Gaussian, F2<:Function} = Message(prod!(msg_2.dist,msg_3.dist))


ruleSPEqualityGaussianGCV(msg_1::Message{F1},msg_2::Message{F2},msg_3::Nothing) where {F1<:Function, F2<:Function} = Message(prod!(msg_1.dist,msg_2.dist))
ruleSPEqualityGaussianGCV(msg_1::Message{F1},msg_2::Nothing,msg_3::Message{F2}) where {F1<:Function, F2<:Function} = Message(prod!(msg_1.dist,msg_3.dist))
ruleSPEqualityGaussianGCV(msg_1::Nothing,msg_2::Message{F2},msg_3::Message{F1}) where {F1<:Function, F2<:Function} = Message(prod!(msg_2.dist,msg_3.dist))

function ruleVBGaussianControlledVarianceOutNDDDD(dist_out::Nothing,
                                                  dist_x::ProbabilityDistribution{Univariate, F},
                                                  dist_z::ProbabilityDistribution{Univariate},
                                                  dist_κ::ProbabilityDistribution{Univariate},
                                                  dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian


    dist_x = convert(ProbabilityDistribution{Univariate,GaussianMeanVariance},dist_x)
    m_x = dist_x.params[:m]
    v_x = dist_x.params[:v]
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    A = exp(-m_ω+v_ω/2)
    B = exp(-m_κ*m_z + ksi/2)


    return Message(Univariate, GaussianMeanVariance, m=m_x, v=inv(A*B))

end

ruleVBGaussianControlledVarianceXDNDDD(dist_out::ProbabilityDistribution{Univariate, F},
                                      dist_x::Nothing,
                                      dist_z::ProbabilityDistribution{Univariate},
                                      dist_κ::ProbabilityDistribution{Univariate},
                                      dist_ω::ProbabilityDistribution{Univariate}) where F<:Gaussian = ruleVBGaussianControlledVarianceOutNDDDD(dist_x,dist_out,dist_z,dist_κ,dist_ω)

function ruleVBGaussianControlledVarianceZDDNDD(dist_out::ProbabilityDistribution{Univariate, F1},
                                      dist_x::ProbabilityDistribution{Univariate, F2},
                                      dist_z::Nothing,
                                      dist_κ::ProbabilityDistribution{Univariate},
                                      dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian, F2<:Gaussian}


    m_out,v_out = unsafeMeanCov(dist_out)
    m_x,v_x = unsafeMeanCov(dist_x)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_ω, v_ω = unsafeMeanCov(dist_ω)

    Psi = (m_out-m_x)^2+v_out+v_x
    A = exp(-m_ω+v_ω/2)
    l_pdf(z) = -0.5*(m_κ*z + Psi*A*exp(-m_κ*z+0.5*v_κ*z^2/2))
    return Message(Univariate, Function, log_pdf=l_pdf)
end

ruleVBGaussianControlledVarianceKDDDND(dist_out::ProbabilityDistribution{Univariate, F1},
                                      dist_x::ProbabilityDistribution{Univariate, F2},
                                      dist_z::ProbabilityDistribution{Univariate},
                                      dist_κ::Nothing,
                                      dist_ω::ProbabilityDistribution{Univariate}) where {F1<:Gaussian, F2<:Gaussian} =
ruleVBGaussianControlledVarianceZDDNDD(dist_out,dist_x,dist_κ,dist_z,dist_ω)

function ruleVBGaussianControlledVarianceΩDDDDN(dist_out::ProbabilityDistribution{Univariate, F1},
                                      dist_x::ProbabilityDistribution{Univariate, F2},
                                      dist_z::ProbabilityDistribution{Univariate},
                                      dist_κ::ProbabilityDistribution{Univariate},
                                      dist_ω::Nothing) where {F1<:Gaussian, F2<:Gaussian}
    m_z, v_z = unsafeMeanCov(dist_z)
    m_κ, v_κ = unsafeMeanCov(dist_κ)
    m_out,v_out = unsafeMeanCov(dist_out)
    m_x,v_x = unsafeMeanCov(dist_x)
    Psi = (m_out-m_x)^2+v_out+v_x
    ksi = m_κ^2*v_z + m_z^2*v_κ+v_z*v_κ
    B = exp(-m_κ*m_z + ksi/2)
    l_pdf(ω) = -0.5*(ω + Psi*B*exp(-ω))
    return Message(Univariate, Function, log_pdf = l_pdf)
end

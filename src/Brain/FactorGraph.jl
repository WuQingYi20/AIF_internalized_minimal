"""
    FactorGraphs

RxInfer-based generative models for agent cognition.
Defines two competing models:
- M₀ (label_blind): Labels don't predict behavior
- M₁ (label_aware): Labels predict behavior (institutional model)
"""
module FactorGraphs

using RxInfer
using Distributions
using SpecialFunctions: loggamma

export label_blind_model, label_aware_model,
       compute_model_evidence_M0, compute_model_evidence_M1,
       compute_free_energy

# M₀: Label-irrelevant generative model.
# Assumes a single global cooperation rate θ that doesn't depend on labels.
# P(observations | θ) = ∏ᵢ Bernoulli(oᵢ | θ)
# P(θ) = Beta(α, β)
@model function label_blind_model(observations, prior_α, prior_β)
    # Prior on global cooperation rate
    θ ~ Beta(prior_α, prior_β)

    # Likelihood: all observations come from same distribution
    for i in eachindex(observations)
        observations[i] ~ Bernoulli(θ)
    end
end

# M₁: Label-dependent generative model (institutional).
# Assumes separate cooperation rates for ingroup (θ_in) and outgroup (θ_out).
# P(observations | labels, θ_in, θ_out) = ∏ᵢ Bernoulli(oᵢ | θ_labels[i])
# P(θ_in) = Beta(α_in, β_in)
# P(θ_out) = Beta(α_out, β_out)
@model function label_aware_model(observations, labels, prior_α_in, prior_β_in,
                                  prior_α_out, prior_β_out)
    # Priors on ingroup and outgroup cooperation rates
    θ_in ~ Beta(prior_α_in, prior_β_in)
    θ_out ~ Beta(prior_α_out, prior_β_out)

    # Likelihood: observations depend on label
    for i in eachindex(observations)
        # labels[i] == true means ingroup
        θ_effective = labels[i] ? θ_in : θ_out
        observations[i] ~ Bernoulli(θ_effective)
    end
end

# Simplified label-aware model where we split data by label first.
# This avoids the indexing issue in the model definition.
@model function label_aware_model_split(ingroup_obs, outgroup_obs,
                                       prior_α_in, prior_β_in,
                                       prior_α_out, prior_β_out)
    # Priors
    θ_in ~ Beta(prior_α_in, prior_β_in)
    θ_out ~ Beta(prior_α_out, prior_β_out)

    # Ingroup observations
    for i in eachindex(ingroup_obs)
        ingroup_obs[i] ~ Bernoulli(θ_in)
    end

    # Outgroup observations
    for i in eachindex(outgroup_obs)
        outgroup_obs[i] ~ Bernoulli(θ_out)
    end
end

"""
    compute_model_evidence_M0(observations, prior_α, prior_β) -> Float64

Compute log model evidence (negative free energy) for M₀.
Uses analytical solution for Beta-Bernoulli conjugate pair.
"""
function compute_model_evidence_M0(observations::AbstractVector{Bool},
                                   prior_α::Float64, prior_β::Float64)
    if isempty(observations)
        return 0.0  # No evidence for empty data
    end

    n = length(observations)
    k = count(observations)  # Number of successes (cooperations)

    # Analytical log evidence for Beta-Bernoulli:
    # log P(D|M₀) = log B(α + k, β + n - k) - log B(α, β)
    # where B is the beta function

    # Using log gamma for numerical stability
    log_evidence = (
        loggamma(prior_α + prior_β) - loggamma(prior_α) - loggamma(prior_β) +
        loggamma(prior_α + k) + loggamma(prior_β + n - k) -
        loggamma(prior_α + prior_β + n)
    )

    return log_evidence
end

"""
    compute_model_evidence_M1(observations, labels, priors...) -> Float64

Compute log model evidence for M₁ (label-aware model).
Splits data by label and computes evidence for each group.
"""
function compute_model_evidence_M1(observations::AbstractVector{Bool},
                                   labels::AbstractVector{Bool},
                                   prior_α_in::Float64, prior_β_in::Float64,
                                   prior_α_out::Float64, prior_β_out::Float64)
    if isempty(observations)
        return 0.0
    end

    # Split observations by label
    ingroup_obs = observations[labels]
    outgroup_obs = observations[.!labels]

    # Compute evidence for each group independently
    log_evidence_in = if !isempty(ingroup_obs)
        compute_model_evidence_M0(ingroup_obs, prior_α_in, prior_β_in)
    else
        0.0
    end

    log_evidence_out = if !isempty(outgroup_obs)
        compute_model_evidence_M0(outgroup_obs, prior_α_out, prior_β_out)
    else
        0.0
    end

    # Total evidence is product (sum in log space)
    return log_evidence_in + log_evidence_out
end

"""
    compute_free_energy(model_type, data...) -> Float64

Compute variational free energy (negative log evidence approximation).
Lower is better.
"""
function compute_free_energy(model_type::Symbol, data...)
    if model_type == :neutral
        observations, prior_α, prior_β = data
        return -compute_model_evidence_M0(observations, prior_α, prior_β)
    elseif model_type == :institutional
        observations, labels, prior_α_in, prior_β_in, prior_α_out, prior_β_out = data
        return -compute_model_evidence_M1(observations, labels,
                                          prior_α_in, prior_β_in,
                                          prior_α_out, prior_β_out)
    else
        error("Unknown model type: $model_type")
    end
end

"""
Run inference using RxInfer for more complex models.
This is the full variational inference approach.
"""
function run_rxinfer_inference_M0(observations::AbstractVector{Bool},
                                  prior_α::Float64, prior_β::Float64;
                                  iterations::Int=10)
    if isempty(observations)
        return (
            posterior_θ = Beta(prior_α, prior_β),
            free_energy = 0.0
        )
    end

    result = infer(
        model = label_blind_model(prior_α=prior_α, prior_β=prior_β),
        data = (observations = observations,),
        free_energy = true,
        iterations = iterations
    )

    return (
        posterior_θ = result.posteriors[:θ],
        free_energy = isnothing(result.free_energy) ? Inf : last(result.free_energy)
    )
end

"""
Run inference for split label-aware model.
"""
function run_rxinfer_inference_M1(ingroup_obs::AbstractVector{Bool},
                                  outgroup_obs::AbstractVector{Bool},
                                  prior_α_in::Float64, prior_β_in::Float64,
                                  prior_α_out::Float64, prior_β_out::Float64;
                                  iterations::Int=10)
    if isempty(ingroup_obs) && isempty(outgroup_obs)
        return (
            posterior_θ_in = Beta(prior_α_in, prior_β_in),
            posterior_θ_out = Beta(prior_α_out, prior_β_out),
            free_energy = 0.0
        )
    end

    result = infer(
        model = label_aware_model_split(
            prior_α_in=prior_α_in, prior_β_in=prior_β_in,
            prior_α_out=prior_α_out, prior_β_out=prior_β_out
        ),
        data = (
            ingroup_obs = isempty(ingroup_obs) ? missing : ingroup_obs,
            outgroup_obs = isempty(outgroup_obs) ? missing : outgroup_obs
        ),
        free_energy = true,
        iterations = iterations
    )

    return (
        posterior_θ_in = result.posteriors[:θ_in],
        posterior_θ_out = result.posteriors[:θ_out],
        free_energy = isnothing(result.free_energy) ? Inf : last(result.free_energy)
    )
end

end # module

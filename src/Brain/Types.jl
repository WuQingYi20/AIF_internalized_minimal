"""
    BrainTypes

Type definitions for agent cognitive states and internal models.
"""
module BrainTypes

using Distributions

export CognitiveState, ActiveModel, NEUTRAL, INSTITUTIONAL,
       BeliefState, PreferredOutcome,
       predict_cooperation, global_cooperation_mean,
       ingroup_cooperation_mean, outgroup_cooperation_mean

"""
    ActiveModel

Enumeration of generative model structures an agent can use.

- `NEUTRAL`: Label-irrelevant model (M₀) - assumes labels don't predict behavior
- `INSTITUTIONAL`: Label-aware model (M₁) - assumes labels predict behavior
"""
@enum ActiveModel begin
    NEUTRAL        # M₀: P(Action | Label) = P(Action)
    INSTITUTIONAL  # M₁: P(Action | Label) ≠ P(Action)
end

"""
    BeliefState

Posterior beliefs about cooperation rates.

For NEUTRAL model: single θ estimate
For INSTITUTIONAL model: separate θ_in and θ_out estimates
"""
mutable struct BeliefState
    # Global cooperation rate (for M₀)
    α_global::Float64  # Beta posterior alpha
    β_global::Float64  # Beta posterior beta

    # Ingroup cooperation rate (for M₁)
    α_ingroup::Float64
    β_ingroup::Float64

    # Outgroup cooperation rate (for M₁)
    α_outgroup::Float64
    β_outgroup::Float64

    function BeliefState(; prior::Tuple{Float64,Float64}=(1.0, 1.0))
        new(prior[1], prior[2],  # global
            prior[1], prior[2],  # ingroup
            prior[1], prior[2])  # outgroup
    end
end

"""
Get the mean cooperation rate estimate for global model.
"""
function global_cooperation_mean(b::BeliefState)
    b.α_global / (b.α_global + b.β_global)
end

"""
Get the mean cooperation rate estimate for ingroup.
"""
function ingroup_cooperation_mean(b::BeliefState)
    b.α_ingroup / (b.α_ingroup + b.β_ingroup)
end

"""
Get the mean cooperation rate estimate for outgroup.
"""
function outgroup_cooperation_mean(b::BeliefState)
    b.α_outgroup / (b.α_outgroup + b.β_outgroup)
end

"""
    PreferredOutcome

Agent's preferences over interaction outcomes.
Used for computing risk component of Expected Free Energy.
"""
struct PreferredOutcome
    prefer_cooperation::Float64     # Preference weight for mutual cooperation
    prefer_exploitation::Float64    # Preference weight for exploiting defector
    prefer_being_exploited::Float64 # Preference weight for being exploited
    prefer_mutual_defection::Float64 # Preference weight for mutual defection

    function PreferredOutcome(;
        cooperation::Float64 = 0.7,
        exploitation::Float64 = 0.1,
        being_exploited::Float64 = 0.05,
        mutual_defection::Float64 = 0.15
    )
        # Normalize to probability distribution
        total = cooperation + exploitation + being_exploited + mutual_defection
        new(
            cooperation / total,
            exploitation / total,
            being_exploited / total,
            mutual_defection / total
        )
    end
end

"""
    CognitiveState

Complete cognitive state of an agent, including beliefs, model selection state,
and internalization parameters.

# Fields
- `beliefs::BeliefState`: Posterior beliefs about cooperation rates
- `active_model::ActiveModel`: Currently active generative model
- `γ::Float64`: Precision/confidence in institutional prior (internalization depth)
- `action_precision::Float64`: β for softmax action selection
- `preferred_outcomes::PreferredOutcome`: Outcome preferences for EFE
- `model_evidence::Dict{ActiveModel,Float64}`: Accumulated log evidence for each model
"""
mutable struct CognitiveState
    beliefs::BeliefState
    active_model::ActiveModel
    γ::Float64  # Internalization precision
    action_precision::Float64  # Action selection temperature
    preferred_outcomes::PreferredOutcome
    model_evidence::Dict{ActiveModel,Float64}

    function CognitiveState(;
        prior_cooperation::Tuple{Float64,Float64} = (1.0, 1.0),
        γ::Float64 = 1.0,
        action_precision::Float64 = 2.0
    )
        new(
            BeliefState(prior=prior_cooperation),
            NEUTRAL,  # Start with neutral model
            γ,
            action_precision,
            PreferredOutcome(),
            Dict(NEUTRAL => 0.0, INSTITUTIONAL => 0.0)
        )
    end
end

"""
Get predicted cooperation probability for an opponent based on current model.
"""
function predict_cooperation(state::CognitiveState, is_ingroup::Bool)
    if state.active_model == NEUTRAL
        return global_cooperation_mean(state.beliefs)
    else
        # Use institutional model with internalization weighting
        base_rate = global_cooperation_mean(state.beliefs)
        if is_ingroup
            institutional_rate = ingroup_cooperation_mean(state.beliefs)
        else
            institutional_rate = outgroup_cooperation_mean(state.beliefs)
        end
        # Interpolate based on γ (higher γ = more weight on institutional belief)
        weight = state.γ / (1.0 + state.γ)
        return weight * institutional_rate + (1 - weight) * base_rate
    end
end

"""
Update beliefs after observing an interaction outcome.
"""
function update_beliefs!(state::CognitiveState, opponent_label::Bool,
                        own_label::Bool, opponent_cooperated::Bool)
    # Update global belief
    if opponent_cooperated
        state.beliefs.α_global += 1.0
    else
        state.beliefs.β_global += 1.0
    end

    # Update label-specific beliefs
    is_ingroup = opponent_label == own_label
    if is_ingroup
        if opponent_cooperated
            state.beliefs.α_ingroup += 1.0
        else
            state.beliefs.β_ingroup += 1.0
        end
    else
        if opponent_cooperated
            state.beliefs.α_outgroup += 1.0
        else
            state.beliefs.β_outgroup += 1.0
        end
    end
end

end # module

"""
    Learning

Structure learning and parameter update logic.
Implements model comparison via free energy and internalization dynamics.
"""
module Learning

using ..BrainTypes
using ..WorldTypes
using ..FactorGraphs
using Statistics

export compute_model_comparison, structure_learning_decision,
       update_beliefs!, update_internalization!, maybe_switch_model!

"""
    compute_model_comparison(agent, own_label) -> (evidence_M0, evidence_M1)

Compare evidence for neutral (M₀) vs institutional (M₁) models.
Returns log evidence for each model.
"""
function compute_model_comparison(agent::InstitutionAgent, own_label::Bool)
    history = agent.interaction_history
    state = agent.cognitive_state

    if isempty(history)
        return (0.0, 0.0)
    end

    # Extract observations and labels from history
    observations = [r.opponent_cooperated for r in history]
    labels = [r.opponent_label == own_label for r in history]  # true = ingroup

    # Get current priors from belief state
    beliefs = state.beliefs

    # Evidence for M₀ (global model)
    evidence_M0 = compute_model_evidence_M0(
        observations,
        beliefs.α_global,
        beliefs.β_global
    )

    # Evidence for M₁ (label-aware model)
    evidence_M1 = compute_model_evidence_M1(
        observations,
        labels,
        beliefs.α_ingroup, beliefs.β_ingroup,
        beliefs.α_outgroup, beliefs.β_outgroup
    )

    return (evidence_M0, evidence_M1)
end

"""
    structure_learning_decision(agent, own_label, complexity_penalty) -> ActiveModel

Decide which model to use based on evidence and complexity penalty.
Returns the model with higher adjusted evidence.
"""
function structure_learning_decision(agent::InstitutionAgent, own_label::Bool,
                                     complexity_penalty::Float64)::ActiveModel
    evidence_M0, evidence_M1 = compute_model_comparison(agent, own_label)

    # Apply complexity penalty to M₁ (has more parameters)
    # This implements Occam's razor: simpler model wins when evidence is similar
    adjusted_M1 = evidence_M1 - complexity_penalty

    # Accumulate evidence in cognitive state
    state = agent.cognitive_state
    state.model_evidence[NEUTRAL] = evidence_M0
    state.model_evidence[INSTITUTIONAL] = adjusted_M1

    # Model selection: choose model with higher evidence
    return adjusted_M1 > evidence_M0 ? INSTITUTIONAL : NEUTRAL
end

"""
    maybe_switch_model!(agent, config)

Check if agent should switch models based on accumulated evidence.
Only switches if there's substantial evidence difference.
"""
function maybe_switch_model!(agent::InstitutionAgent, config)
    current_model = agent.cognitive_state.active_model
    new_model = structure_learning_decision(
        agent, agent.label, config.complexity_penalty
    )

    # Calculate evidence difference
    evidence_diff = abs(
        agent.cognitive_state.model_evidence[INSTITUTIONAL] -
        agent.cognitive_state.model_evidence[NEUTRAL]
    )

    # Only switch if evidence difference is substantial (log Bayes factor > 1)
    if new_model != current_model && evidence_diff > 1.0
        agent.cognitive_state.active_model = new_model
    end
end

"""
    update_beliefs!(agent, record, config)

Update agent's beliefs after observing an interaction outcome.
Updates both global and label-specific beliefs.
"""
function update_beliefs!(agent::InstitutionAgent, record::InteractionRecord, config)
    state = agent.cognitive_state
    beliefs = state.beliefs

    opponent_cooperated = record.opponent_cooperated
    is_ingroup = record.opponent_label == agent.label

    # Update global belief (for M₀)
    if opponent_cooperated
        beliefs.α_global += 1.0
    else
        beliefs.β_global += 1.0
    end

    # Update label-specific beliefs (for M₁)
    if is_ingroup
        if opponent_cooperated
            beliefs.α_ingroup += 1.0
        else
            beliefs.β_ingroup += 1.0
        end
    else
        if opponent_cooperated
            beliefs.α_outgroup += 1.0
        else
            beliefs.β_outgroup += 1.0
        end
    end
end

"""
    update_internalization!(agent, record, config)

Update internalization depth (γ) based on whether outcome matches expectation.
Successful predictions strengthen the institutional prior.
"""
function update_internalization!(agent::InstitutionAgent, record::InteractionRecord,
                                 config)
    state = agent.cognitive_state

    # Only update internalization when using institutional model
    if state.active_model != INSTITUTIONAL
        return
    end

    is_ingroup = record.opponent_label == agent.label
    opponent_cooperated = record.opponent_cooperated

    # Get expected cooperation rate under institutional model
    predicted_rate = predict_cooperation(state, is_ingroup)

    # Did outcome match expectation?
    # High predicted cooperation + actual cooperation = match
    # Low predicted cooperation + actual defection = match
    expected_cooperation = predicted_rate > 0.5
    outcome_matches = expected_cooperation == opponent_cooperated

    # Update γ based on prediction success
    if outcome_matches
        # Strengthen internalization (prediction confirmed)
        state.γ = min(state.γ * 1.1, config.max_precision)
    else
        # Weaken internalization (prediction violated)
        state.γ = max(state.γ * 0.95, config.min_precision)
    end
end

"""
    compute_surprise(agent, record) -> Float64

Compute surprise (negative log likelihood) for an observation.
Used for free energy calculation.
"""
function compute_surprise(agent::InstitutionAgent, record::InteractionRecord)
    state = agent.cognitive_state
    is_ingroup = record.opponent_label == agent.label

    # Get predicted cooperation probability
    p_cooperate = predict_cooperation(state, is_ingroup)

    # Compute negative log likelihood
    if record.opponent_cooperated
        return -log(max(p_cooperate, 1e-10))
    else
        return -log(max(1 - p_cooperate, 1e-10))
    end
end

"""
    compute_information_gain(agent, own_label) -> Float64

Compute information gain (epistemic value) from switching to institutional model.
Measures reduction in uncertainty about opponent behavior.
"""
function compute_information_gain(agent::InstitutionAgent, own_label::Bool)
    state = agent.cognitive_state
    beliefs = state.beliefs

    # Entropy under global model
    global_mean = beliefs.α_global / (beliefs.α_global + beliefs.β_global)
    H_global = -global_mean * log(max(global_mean, 1e-10)) -
               (1 - global_mean) * log(max(1 - global_mean, 1e-10))

    # Entropy under institutional model (average of ingroup and outgroup)
    in_mean = beliefs.α_ingroup / (beliefs.α_ingroup + beliefs.β_ingroup)
    out_mean = beliefs.α_outgroup / (beliefs.α_outgroup + beliefs.β_outgroup)

    H_ingroup = -in_mean * log(max(in_mean, 1e-10)) -
                (1 - in_mean) * log(max(1 - in_mean, 1e-10))
    H_outgroup = -out_mean * log(max(out_mean, 1e-10)) -
                 (1 - out_mean) * log(max(1 - out_mean, 1e-10))

    # Information gain is reduction in expected entropy
    # Weight by empirical frequency of ingroup vs outgroup interactions
    history = agent.interaction_history
    if isempty(history)
        p_ingroup = 0.5
    else
        p_ingroup = count(r -> r.opponent_label == own_label, history) / length(history)
    end

    H_institutional = p_ingroup * H_ingroup + (1 - p_ingroup) * H_outgroup

    return max(H_global - H_institutional, 0.0)  # Information gain is non-negative
end

end # module

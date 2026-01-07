"""
    ActionSelection

Action selection via Expected Free Energy (EFE) minimization.
Implements the core Active Inference decision-making mechanism.

EFE combines:
- Pragmatic value: Expected payoff from the game (using actual payoff matrix)
- Epistemic value: Reduction in uncertainty (ambiguity term)
- Reciprocity value: Preference to match expected opponent behavior (NEW!)

The reciprocity term is key for institutional emergence:
- Without it: defection is always dominant in PD regardless of predictions
- With it: agents cooperate with those they expect to cooperate
- This enables self-fulfilling prophecy: belief → action → confirmation
"""
module ActionSelection

using ..BrainTypes
using ..WorldTypes
using ..Physics: GameType, get_payoff_matrix
using Statistics

export select_action, compute_expected_free_energy, compute_expected_free_energy_with_payoffs,
       softmax, entropy, predict_opponent_action, get_action_probabilities,
       compute_reciprocity_value

"""
    softmax(values, β) -> Vector{Float64}

Compute softmax probabilities with inverse temperature β.
Higher β = more deterministic (greedy), lower β = more exploratory.
"""
function softmax(values::Vector{Float64}, β::Float64)
    # Numerical stability: subtract max
    max_val = maximum(values)
    exp_vals = exp.(β .* (values .- max_val))
    return exp_vals ./ sum(exp_vals)
end

"""
    entropy(p) -> Float64

Compute entropy of a Bernoulli distribution.
H(p) = -p log(p) - (1-p) log(1-p)
"""
function entropy(p::Float64)
    if p ≤ 0 || p ≥ 1
        return 0.0
    end
    return -p * log(p) - (1 - p) * log(1 - p)
end

"""
    predict_opponent_action(agent, opponent_label) -> Float64

Predict probability that opponent will cooperate, given their label.
Uses the agent's current model (NEUTRAL or INSTITUTIONAL).
"""
function predict_opponent_action(agent::InstitutionAgent, opponent_label::Bool)
    state = agent.cognitive_state
    is_ingroup = opponent_label == agent.label
    return predict_cooperation(state, is_ingroup)
end

"""
    compute_reciprocity_value(action, p_opponent_cooperate) -> Float64

Compute reciprocity value: how much the agent values cooperating with expected cooperators.

This is ASYMMETRIC reciprocity (not symmetric matching):
- Only COOPERATION gets a reciprocity bonus
- The bonus scales with expected opponent cooperation
- Defection gets no bonus (neutral)

Why asymmetric? In symmetric reciprocity:
- If p < 0.5, reciprocity favors defection → universal defection equilibrium
- Institutions can't emerge in PD because defection is always stable

With asymmetric reciprocity:
- Cooperation gets a bonus when opponent is expected to cooperate
- This creates potential for differentiated behavior based on predictions
- If you believe ingroup cooperates more → you cooperate more with ingroup

This is the key mechanism for self-fulfilling prophecy:
- Belief about ingroup cooperation → more cooperation with ingroup →
  ingroup actually cooperates more → belief confirmed
"""
function compute_reciprocity_value(action::Bool, p_opponent_cooperate::Float64)
    if action  # I'm cooperating
        # Cooperation bonus = expected opponent cooperation
        # Higher if I believe opponent will cooperate
        return p_opponent_cooperate
    else  # I'm defecting
        # Defection gets no bonus - it's the "default" rational choice in PD
        # This asymmetry is key for enabling cooperation-based institutions
        return 0.0
    end
end

"""
    compute_expected_free_energy(agent, action, opponent_label, game) -> Float64

Compute Expected Free Energy (G) for a given action.

G(π) = -E[Payoff] + Ambiguity - γ × Reciprocity

Where:
- Payoff: Expected payoff from the game's payoff matrix (pragmatic value)
- Ambiguity: Entropy of predicted opponent action (epistemic uncertainty)
- Reciprocity: Alignment between my action and expected opponent action (NEW!)

The reciprocity term is weighted by γ (internalization depth):
- Higher γ → more weight on reciprocity → stronger self-fulfilling prophecy
- This creates feedback: beliefs → actions → confirmation → stronger beliefs

Lower G is better - agent prefers actions that:
1. Lead to high expected payoffs (pragmatic value)
2. Are taken when opponent behavior is predictable (low ambiguity)
3. Match the expected behavior of the opponent (reciprocity)
"""
function compute_expected_free_energy(agent::InstitutionAgent,
                                      action::Bool,
                                      opponent_label::Bool,
                                      game::GameType)
    state = agent.cognitive_state
    p_opponent_cooperate = predict_opponent_action(agent, opponent_label)

    # Get actual payoff matrix for this game
    payoffs = get_payoff_matrix(game)
    # payoffs[1,1] = CC, payoffs[1,2] = CD, payoffs[2,1] = DC, payoffs[2,2] = DD

    # === PRAGMATIC VALUE: Expected payoff ===
    if action  # Cooperating (row 1)
        expected_value = p_opponent_cooperate * payoffs[1,1] + (1 - p_opponent_cooperate) * payoffs[1,2]
    else  # Defecting (row 2)
        expected_value = p_opponent_cooperate * payoffs[2,1] + (1 - p_opponent_cooperate) * payoffs[2,2]
    end

    # === EPISTEMIC VALUE: Ambiguity ===
    # Higher entropy = less predictable = higher EFE (worse)
    ambiguity = entropy(p_opponent_cooperate)

    # === RECIPROCITY VALUE: Action-belief alignment ===
    # This is the key for making cognitive beliefs affect behavior!
    # Weight by γ: internalized agents reciprocate more strongly
    reciprocity = compute_reciprocity_value(action, p_opponent_cooperate)

    # Only apply reciprocity when using institutional model
    # (NEUTRAL agents don't differentiate, so reciprocity has no differential effect)
    if state.active_model == INSTITUTIONAL
        reciprocity_weight = state.γ
    else
        reciprocity_weight = 0.0
    end

    # EFE = -expected_value + ambiguity - reciprocity_weight * reciprocity
    # Lower is better: high value, low ambiguity, high reciprocity
    return -expected_value + ambiguity - reciprocity_weight * reciprocity
end

# Alias for backward compatibility
compute_expected_free_energy_with_payoffs = compute_expected_free_energy

"""
    select_action(agent, opponent_label, config) -> Bool

Select action (cooperate=true, defect=false) using Expected Free Energy.
Uses softmax policy over negative EFE (lower G = higher probability).
"""
function select_action(agent::InstitutionAgent, opponent_label::Bool, config)::Bool
    game = config.game_type
    G_cooperate = compute_expected_free_energy(agent, true, opponent_label, game)
    G_defect = compute_expected_free_energy(agent, false, opponent_label, game)

    # Convert to policy via softmax over negative EFE (lower G = higher probability)
    β = agent.cognitive_state.action_precision
    policy = softmax([-G_cooperate, -G_defect], β)

    p_cooperate = policy[1]

    # Sample action from policy
    return rand() < p_cooperate
end

"""
    select_action_deterministic(agent, opponent_label, game) -> Bool

Select action deterministically (greedy w.r.t. EFE).
Useful for analysis.
"""
function select_action_deterministic(agent::InstitutionAgent, opponent_label::Bool,
                                     game::GameType)::Bool
    G_cooperate = compute_expected_free_energy(agent, true, opponent_label, game)
    G_defect = compute_expected_free_energy(agent, false, opponent_label, game)

    return G_cooperate < G_defect
end

"""
    get_action_probabilities(agent, opponent_label, config) -> Tuple{Float64, Float64}

Get probabilities for cooperate and defect actions.
"""
function get_action_probabilities(agent::InstitutionAgent, opponent_label::Bool, config)
    game = config.game_type
    G_cooperate = compute_expected_free_energy(agent, true, opponent_label, game)
    G_defect = compute_expected_free_energy(agent, false, opponent_label, game)

    β = agent.cognitive_state.action_precision
    policy = softmax([-G_cooperate, -G_defect], β)

    return (policy[1], policy[2])  # (p_cooperate, p_defect)
end

end # module

"""
    ActionSelection

Action selection via Expected Free Energy (EFE) minimization.
Implements the core Active Inference decision-making mechanism.
"""
module ActionSelection

using ..BrainTypes
using ..WorldTypes
using Statistics

export select_action, compute_expected_free_energy,
       softmax, entropy, kl_divergence

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

Compute entropy of a probability distribution.
H(p) = -∑ p(x) log p(x)
"""
function entropy(p::Float64)
    # Entropy of Bernoulli distribution
    if p ≤ 0 || p ≥ 1
        return 0.0
    end
    return -p * log(p) - (1 - p) * log(1 - p)
end

function entropy(p::Vector{Float64})
    # Entropy of categorical distribution
    h = 0.0
    for pi in p
        if pi > 0
            h -= pi * log(pi)
        end
    end
    return h
end

"""
    kl_divergence(p, q) -> Float64

Compute KL divergence D_KL(p || q) for Bernoulli distributions.
"""
function kl_divergence(p::Float64, q::Float64)
    # Clamp to avoid numerical issues
    p = clamp(p, 1e-10, 1 - 1e-10)
    q = clamp(q, 1e-10, 1 - 1e-10)

    return p * log(p / q) + (1 - p) * log((1 - p) / (1 - q))
end

function kl_divergence(p::Vector{Float64}, q::Vector{Float64})
    # KL divergence for categorical distributions
    kl = 0.0
    for (pi, qi) in zip(p, q)
        if pi > 0
            qi_safe = max(qi, 1e-10)
            kl += pi * log(pi / qi_safe)
        end
    end
    return kl
end

"""
    predict_opponent_action(agent, opponent_label) -> Float64

Predict probability that opponent will cooperate, given their label.
"""
function predict_opponent_action(agent::InstitutionAgent, opponent_label::Bool)
    state = agent.cognitive_state
    is_ingroup = opponent_label == agent.label
    return predict_cooperation(state, is_ingroup)
end

"""
    predict_outcome_distribution(agent, own_action, opponent_label) -> Vector{Float64}

Predict distribution over outcomes given own action and opponent label.
Returns [P(CC), P(CD), P(DC), P(DD)] where first letter is own action.
"""
function predict_outcome_distribution(agent::InstitutionAgent,
                                      own_action::Bool,
                                      opponent_label::Bool)
    p_opponent_cooperate = predict_opponent_action(agent, opponent_label)

    if own_action  # Agent cooperates
        # Outcomes: CC (opponent cooperates) or CD (opponent defects)
        return [p_opponent_cooperate, 1 - p_opponent_cooperate, 0.0, 0.0]
    else  # Agent defects
        # Outcomes: DC (opponent cooperates) or DD (opponent defects)
        return [0.0, 0.0, p_opponent_cooperate, 1 - p_opponent_cooperate]
    end
end

"""
    get_preferred_outcome_distribution(agent) -> Vector{Float64}

Get agent's preferred distribution over outcomes.
Returns [P(CC), P(CD), P(DC), P(DD)].
"""
function get_preferred_outcome_distribution(agent::InstitutionAgent)
    prefs = agent.cognitive_state.preferred_outcomes
    return [
        prefs.prefer_cooperation,        # CC
        prefs.prefer_being_exploited,    # CD (agent cooperates, opponent defects)
        prefs.prefer_exploitation,       # DC (agent defects, opponent cooperates)
        prefs.prefer_mutual_defection    # DD
    ]
end

"""
    compute_expected_free_energy(agent, action, opponent_label) -> Float64

Compute Expected Free Energy (G) for a given action.

G(π) = Ambiguity + Risk
     = E_Q[H[P(o|s)]] + D_KL[Q(s|π) || P(s)]

Where:
- Ambiguity: expected entropy of observations (uncertainty about outcomes)
- Risk: divergence from preferred outcomes (goal-directedness)

Lower G is better - agent prefers actions that:
1. Lead to predictable outcomes (low ambiguity)
2. Lead to preferred outcomes (low risk)
"""
function compute_expected_free_energy(agent::InstitutionAgent,
                                      action::Bool,
                                      opponent_label::Bool)
    # Predict outcome distribution under this action
    q_outcome = predict_outcome_distribution(agent, action, opponent_label)

    # Get preferred outcome distribution
    preferred = get_preferred_outcome_distribution(agent)

    # AMBIGUITY: Entropy of predicted observations
    # Higher entropy = less predictable = worse
    p_opponent_cooperate = predict_opponent_action(agent, opponent_label)
    ambiguity = entropy(p_opponent_cooperate)

    # RISK: KL divergence from preferred outcomes
    # Higher divergence = further from goals = worse
    # Only consider outcomes possible under this action
    if action  # Cooperating
        # Possible outcomes: CC or CD
        q_relevant = [q_outcome[1], q_outcome[2]]
        p_relevant = [preferred[1], preferred[2]]
        # Normalize
        q_sum = sum(q_relevant)
        p_sum = sum(p_relevant)
        if q_sum > 0 && p_sum > 0
            q_norm = q_relevant ./ q_sum
            p_norm = p_relevant ./ p_sum
            risk = kl_divergence(q_norm, p_norm)
        else
            risk = 0.0
        end
    else  # Defecting
        # Possible outcomes: DC or DD
        q_relevant = [q_outcome[3], q_outcome[4]]
        p_relevant = [preferred[3], preferred[4]]
        q_sum = sum(q_relevant)
        p_sum = sum(p_relevant)
        if q_sum > 0 && p_sum > 0
            q_norm = q_relevant ./ q_sum
            p_norm = p_relevant ./ p_sum
            risk = kl_divergence(q_norm, p_norm)
        else
            risk = 0.0
        end
    end

    # Total EFE
    return ambiguity + risk
end

"""
    compute_expected_free_energy_simple(agent, action, opponent_label) -> Float64

Simplified EFE computation based on expected payoff and uncertainty.
More interpretable but less theoretically grounded.
"""
function compute_expected_free_energy_simple(agent::InstitutionAgent,
                                             action::Bool,
                                             opponent_label::Bool)
    p_opponent_cooperate = predict_opponent_action(agent, opponent_label)

    # Expected "value" under this action (higher is better, so negate for EFE)
    if action  # Cooperating
        # Payoff if opponent cooperates (CC): 3, if defects (CD): 0
        expected_value = p_opponent_cooperate * 3.0 + (1 - p_opponent_cooperate) * 0.0
    else  # Defecting
        # Payoff if opponent cooperates (DC): 5, if defects (DD): 1
        expected_value = p_opponent_cooperate * 5.0 + (1 - p_opponent_cooperate) * 1.0
    end

    # Ambiguity term
    ambiguity = entropy(p_opponent_cooperate)

    # Negate value (since lower EFE is better) and add ambiguity
    return -expected_value + ambiguity
end

"""
    select_action(agent, opponent_label, config) -> Bool

Select action (cooperate=true, defect=false) using Expected Free Energy.
"""
function select_action(agent::InstitutionAgent, opponent_label::Bool, config)::Bool
    # Compute EFE for each action
    G_cooperate = compute_expected_free_energy(agent, true, opponent_label)
    G_defect = compute_expected_free_energy(agent, false, opponent_label)

    # Convert to policy via softmax over negative EFE (lower G = higher probability)
    β = agent.cognitive_state.action_precision
    policy = softmax([-G_cooperate, -G_defect], β)

    p_cooperate = policy[1]

    # Sample action from policy
    return rand() < p_cooperate
end

"""
    select_action_deterministic(agent, opponent_label) -> Bool

Select action deterministically (greedy w.r.t. EFE).
Useful for analysis.
"""
function select_action_deterministic(agent::InstitutionAgent, opponent_label::Bool)::Bool
    G_cooperate = compute_expected_free_energy(agent, true, opponent_label)
    G_defect = compute_expected_free_energy(agent, false, opponent_label)

    return G_cooperate < G_defect
end

"""
    get_action_probabilities(agent, opponent_label, config) -> Tuple{Float64, Float64}

Get probabilities for cooperate and defect actions.
"""
function get_action_probabilities(agent::InstitutionAgent, opponent_label::Bool, config)
    G_cooperate = compute_expected_free_energy(agent, true, opponent_label)
    G_defect = compute_expected_free_energy(agent, false, opponent_label)

    β = agent.cognitive_state.action_precision
    policy = softmax([-G_cooperate, -G_defect], β)

    return (policy[1], policy[2])  # (p_cooperate, p_defect)
end

end # module

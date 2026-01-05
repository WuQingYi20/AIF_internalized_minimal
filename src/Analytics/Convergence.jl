"""
    Convergence

Metrics for measuring institutional emergence and internalization.
"""
module Convergence

using Agents
using Statistics
using ..BrainTypes
using ..WorldTypes

export institutional_adoption_rate, mean_internalization, label_correlation,
       compute_free_energy_trajectory, cooperation_gap,
       compute_consensus_measure, compute_self_fulfilling_index

"""
    institutional_adoption_rate(model_or_sim) -> Float64

Fraction of agents using the institutional (label-aware) model.
Returns value in [0, 1].
"""
function institutional_adoption_rate(sim_or_model)
    # Handle both Simulation type (which has .model) and direct ABM
    model = hasproperty(sim_or_model, :model) ? sim_or_model.model : sim_or_model
    agents = collect(allagents(model))
    n_institutional = count(a -> a.cognitive_state.active_model == INSTITUTIONAL, agents)
    return n_institutional / length(agents)
end

"""
    mean_internalization(model_or_sim) -> Float64

Average internalization depth (γ) across all agents.
Higher values indicate deeper internalization of institutional beliefs.
"""
function mean_internalization(sim_or_model)
    model = hasproperty(sim_or_model, :model) ? sim_or_model.model : sim_or_model
    agents = collect(allagents(model))
    return mean(a.cognitive_state.γ for a in agents)
end

"""
    internalization_distribution(model) -> Vector{Float64}

Get distribution of internalization values across agents.
"""
function internalization_distribution(model)
    return [a.cognitive_state.γ for a in allagents(model)]
end

"""
    label_correlation(model_or_sim) -> Float64

Compute actual correlation between labels and cooperation behavior.
This measures the "self-fulfilling prophecy" effect.

Returns value in [-1, 1]:
- 0: No correlation (labels don't predict behavior)
- Positive: Ingroup favoritism (cooperate more with same label)
- Negative: Outgroup favoritism (cooperate more with different label)
"""
function label_correlation(sim_or_model)
    model = hasproperty(sim_or_model, :model) ? sim_or_model.model : sim_or_model

    # Collect all interaction data
    ingroup_coops = Int[]
    outgroup_coops = Int[]

    for agent in allagents(model)
        for record in agent.interaction_history
            is_ingroup = record.opponent_label == agent.label
            cooperated = record.own_action ? 1 : 0

            if is_ingroup
                push!(ingroup_coops, cooperated)
            else
                push!(outgroup_coops, cooperated)
            end
        end
    end

    # Need sufficient data
    if length(ingroup_coops) < 5 || length(outgroup_coops) < 5
        return 0.0
    end

    # Compute cooperation rates
    ingroup_rate = mean(ingroup_coops)
    outgroup_rate = mean(outgroup_coops)

    # Correlation is difference in rates, normalized
    # Map to [-1, 1] range
    max_possible_diff = 1.0  # When one is 1.0 and other is 0.0
    correlation = (ingroup_rate - outgroup_rate) / max_possible_diff

    return clamp(correlation, -1.0, 1.0)
end

"""
    cooperation_gap(model) -> Float64

Difference between ingroup and outgroup cooperation rates.
Positive = ingroup favoritism, Negative = outgroup favoritism.
"""
function cooperation_gap(model)
    ingroup_total = 0
    ingroup_coop = 0
    outgroup_total = 0
    outgroup_coop = 0

    for agent in allagents(model)
        for record in agent.interaction_history
            is_ingroup = record.opponent_label == agent.label

            if is_ingroup
                ingroup_total += 1
                ingroup_coop += record.own_action ? 1 : 0
            else
                outgroup_total += 1
                outgroup_coop += record.own_action ? 1 : 0
            end
        end
    end

    ingroup_rate = ingroup_total > 0 ? ingroup_coop / ingroup_total : 0.5
    outgroup_rate = outgroup_total > 0 ? outgroup_coop / outgroup_total : 0.5

    return ingroup_rate - outgroup_rate
end

"""
    compute_free_energy_trajectory(sim_or_model) -> Vector{Float64}

Compute average free energy across agents over time.
Lower free energy indicates more confident/stable beliefs.
"""
function compute_free_energy_trajectory(sim_or_model)
    model = hasproperty(sim_or_model, :model) ? sim_or_model.model : sim_or_model

    # This would require tracking FE at each step
    # For now, return estimate based on belief entropy

    fe_values = Float64[]

    for agent in allagents(model)
        beliefs = agent.cognitive_state.beliefs

        # Approximate FE as entropy of beliefs
        α = beliefs.α_global
        β = beliefs.β_global
        mean_θ = α / (α + β)

        # Entropy of Beta distribution (approximation)
        entropy_approx = -mean_θ * log(max(mean_θ, 1e-10)) -
                         (1 - mean_θ) * log(max(1 - mean_θ, 1e-10))

        push!(fe_values, entropy_approx)
    end

    return fe_values
end

"""
    compute_consensus_measure(model) -> Float64

Measure how much agents agree on their beliefs about ingroup/outgroup.
Returns value in [0, 1] where 1 = perfect consensus.
"""
function compute_consensus_measure(model)
    institutional_agents = filter(
        a -> a.cognitive_state.active_model == INSTITUTIONAL,
        collect(allagents(model))
    )

    if length(institutional_agents) < 2
        return 0.0
    end

    # Get each agent's ingroup/outgroup belief difference
    belief_diffs = Float64[]
    for agent in institutional_agents
        beliefs = agent.cognitive_state.beliefs
        in_mean = beliefs.α_ingroup / (beliefs.α_ingroup + beliefs.β_ingroup)
        out_mean = beliefs.α_outgroup / (beliefs.α_outgroup + beliefs.β_outgroup)
        push!(belief_diffs, in_mean - out_mean)
    end

    # Consensus is inverse of variance
    if length(belief_diffs) < 2
        return 1.0
    end

    variance = var(belief_diffs)
    consensus = 1.0 / (1.0 + variance)

    return consensus
end

"""
    compute_self_fulfilling_index(model) -> Float64

Measure the degree to which the institution has become self-fulfilling.
Combines label correlation with institutional adoption rate.
"""
function compute_self_fulfilling_index(model)
    adoption = institutional_adoption_rate(model)
    correlation = abs(label_correlation(model))

    # Self-fulfilling when both adoption is high AND correlation is high
    # Use geometric mean to require both conditions
    return sqrt(adoption * correlation)
end

"""
    agent_summary(agent) -> NamedTuple

Get summary statistics for a single agent.
"""
function agent_summary(agent::InstitutionAgent)
    beliefs = agent.cognitive_state.beliefs
    n_interactions = length(agent.interaction_history)

    return (
        id = agent.id,
        label = agent.label,
        active_model = agent.cognitive_state.active_model,
        internalization = agent.cognitive_state.γ,
        n_interactions = n_interactions,
        avg_payoff = n_interactions > 0 ?
            mean(r.payoff for r in agent.interaction_history) : 0.0,
        cooperation_rate = n_interactions > 0 ?
            mean(r.own_action for r in agent.interaction_history) : 0.5,
        ingroup_coop_rate = empirical_coop_rate(agent, :ingroup),
        outgroup_coop_rate = empirical_coop_rate(agent, :outgroup),
        global_belief = beliefs.α_global / (beliefs.α_global + beliefs.β_global),
        ingroup_belief = beliefs.α_ingroup / (beliefs.α_ingroup + beliefs.β_ingroup),
        outgroup_belief = beliefs.α_outgroup / (beliefs.α_outgroup + beliefs.β_outgroup)
    )
end

function empirical_coop_rate(agent::InstitutionAgent, group::Symbol)
    history = if group == :ingroup
        filter(r -> r.opponent_label == agent.label, agent.interaction_history)
    else
        filter(r -> r.opponent_label != agent.label, agent.interaction_history)
    end

    isempty(history) && return 0.5
    return mean(r.own_action for r in history)
end

"""
    simulation_summary(model) -> NamedTuple

Get summary statistics for entire simulation.
"""
function simulation_summary(model)
    props = abmproperties(model)
    return (
        n_agents = nagents(model),
        current_step = props.current_step[],
        adoption_rate = institutional_adoption_rate(model),
        mean_internalization = mean_internalization(model),
        label_correlation = label_correlation(model),
        cooperation_gap = cooperation_gap(model),
        consensus = compute_consensus_measure(model),
        self_fulfilling_index = compute_self_fulfilling_index(model)
    )
end

end # module

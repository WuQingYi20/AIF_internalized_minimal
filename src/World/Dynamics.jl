"""
    Dynamics

Simulation dynamics and agent interaction logic.
Integrates the Brain modules with the World for agent-based simulation.
"""
module Dynamics

using Agents
using Random
using ..BrainTypes
using ..WorldTypes
using ..Physics
using ..Learning
using ..ActionSelection

export agent_step!, model_step!, random_opponent, execute_paired_interactions

"""
    random_opponent(agent, model) -> InstitutionAgent

Select a random opponent for the given agent.
Excludes the agent itself from selection.
"""
function random_opponent(agent::InstitutionAgent, model)
    candidates = [a for a in allagents(model) if a.id != agent.id]
    return rand(candidates)
end

"""
    get_agent_pairs(model) -> Vector{Tuple{InstitutionAgent, InstitutionAgent}}

Generate random pairings of all agents for one round of interactions.
Each agent is paired exactly once (if even number) or once plus one agent
interacts twice (if odd number).
"""
function get_agent_pairs(model)
    agents = collect(allagents(model))
    shuffle!(agents)

    pairs = Tuple{InstitutionAgent, InstitutionAgent}[]

    for i in 1:2:length(agents)-1
        push!(pairs, (agents[i], agents[i+1]))
    end

    # Handle odd agent by pairing with random other agent
    if isodd(length(agents))
        odd_agent = agents[end]
        partner = rand(agents[1:end-1])
        push!(pairs, (odd_agent, partner))
    end

    return pairs
end

"""
    agent_step!(agent, model)

Execute one step for a single agent.
This is called by Agents.jl's step! function.

Note: For pairwise interactions, use model_step! instead which
handles proper pairing of agents.
"""
function agent_step!(agent::InstitutionAgent, model)
    config = abmproperties(model).config

    # Select random opponent
    opponent = random_opponent(agent, model)

    # Agent selects action based on opponent's label
    action = select_action(agent, opponent.label, config)

    # Opponent selects action based on agent's label
    opponent_action = select_action(opponent, agent.label, config)

    # Environment resolves interaction (label-blind!)
    payoff, opponent_payoff = resolve_interaction(action, opponent_action, config.game_type)

    # Record observation
    record = InteractionRecord(
        opponent_label = opponent.label,
        opponent_cooperated = opponent_action,
        own_action = action,
        payoff = payoff
    )
    push!(agent.interaction_history, record)

    # Update beliefs
    update_beliefs!(agent, record, config)

    # Structure learning check (only after sufficient observations)
    if length(agent.interaction_history) >= config.structure_learning_threshold
        maybe_switch_model!(agent, config)
    end

    # Update internalization
    update_internalization!(agent, record, config)
end

"""
    model_step!(model)

Execute one step for the entire model.
All agents are paired and interact simultaneously.
"""
function model_step!(model)
    props = abmproperties(model)
    config = props.config

    # Generate pairings
    pairs = get_agent_pairs(model)

    # Execute all interactions
    for (agent1, agent2) in pairs
        execute_single_interaction!(agent1, agent2, config)
    end

    # Update model step counter
    props.current_step[] += 1
end

"""
    execute_single_interaction!(agent1, agent2, config)

Execute a single interaction between two agents.
Both agents observe and update based on the outcome.
"""
function execute_single_interaction!(agent1::InstitutionAgent,
                                     agent2::InstitutionAgent,
                                     config)
    # Each agent selects action based on opponent's label
    action1 = select_action(agent1, agent2.label, config)
    action2 = select_action(agent2, agent1.label, config)

    # Environment resolves (LABEL-BLIND!)
    payoff1, payoff2 = resolve_interaction(action1, action2, config.game_type)

    # Create observation records
    record1 = InteractionRecord(
        opponent_label = agent2.label,
        opponent_cooperated = action2,
        own_action = action1,
        payoff = payoff1
    )
    record2 = InteractionRecord(
        opponent_label = agent1.label,
        opponent_cooperated = action1,
        own_action = action2,
        payoff = payoff2
    )

    # Add to histories
    push!(agent1.interaction_history, record1)
    push!(agent2.interaction_history, record2)

    # Update beliefs for both agents
    update_beliefs!(agent1, record1, config)
    update_beliefs!(agent2, record2, config)

    # Structure learning checks
    if length(agent1.interaction_history) >= config.structure_learning_threshold
        maybe_switch_model!(agent1, config)
    end
    if length(agent2.interaction_history) >= config.structure_learning_threshold
        maybe_switch_model!(agent2, config)
    end

    # Update internalization
    update_internalization!(agent1, record1, config)
    update_internalization!(agent2, record2, config)

    return (record1, record2)
end

"""
    execute_paired_interactions(model, n_rounds) -> DataFrame

Run multiple rounds of paired interactions and collect data.
"""
function execute_paired_interactions(model, n_rounds::Int)
    records = []
    props = abmproperties(model)

    for round in 1:n_rounds
        pairs = get_agent_pairs(model)

        for (agent1, agent2) in pairs
            record1, record2 = execute_single_interaction!(agent1, agent2, props.config)

            push!(records, (
                round = round,
                agent_id = agent1.id,
                agent_label = agent1.label,
                opponent_id = agent2.id,
                opponent_label = agent2.label,
                action = record1.own_action,
                opponent_action = record1.opponent_cooperated,
                payoff = record1.payoff,
                active_model = agent1.cognitive_state.active_model,
                internalization = agent1.cognitive_state.γ
            ))

            push!(records, (
                round = round,
                agent_id = agent2.id,
                agent_label = agent2.label,
                opponent_id = agent1.id,
                opponent_label = agent1.label,
                action = record2.own_action,
                opponent_action = record2.opponent_cooperated,
                payoff = record2.payoff,
                active_model = agent2.cognitive_state.active_model,
                internalization = agent2.cognitive_state.γ
            ))
        end

        props.current_step[] += 1
    end

    return records
end

"""
    compute_group_statistics(model)

Compute statistics about ingroup vs outgroup interactions.
"""
function compute_group_statistics(model)
    ingroup_cooperations = 0
    ingroup_total = 0
    outgroup_cooperations = 0
    outgroup_total = 0

    for agent in allagents(model)
        for record in agent.interaction_history
            is_ingroup = record.opponent_label == agent.label

            if is_ingroup
                ingroup_total += 1
                if record.own_action
                    ingroup_cooperations += 1
                end
            else
                outgroup_total += 1
                if record.own_action
                    outgroup_cooperations += 1
                end
            end
        end
    end

    ingroup_rate = ingroup_total > 0 ? ingroup_cooperations / ingroup_total : 0.5
    outgroup_rate = outgroup_total > 0 ? outgroup_cooperations / outgroup_total : 0.5

    return (
        ingroup_cooperation_rate = ingroup_rate,
        outgroup_cooperation_rate = outgroup_rate,
        ingroup_interactions = ingroup_total,
        outgroup_interactions = outgroup_total,
        cooperation_gap = ingroup_rate - outgroup_rate
    )
end

end # module

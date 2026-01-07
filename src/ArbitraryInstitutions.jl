"""
    ArbitraryInstitutions

A Julia framework for simulating institution emergence through Active Inference.
Agents interact in a Minimal Group Paradigm where meaningless labels become
self-fulfilling prophecies through structure learning and belief internalization.
"""
module ArbitraryInstitutions

using Random
using Statistics
using Distributions
using DataFrames
using UUIDs

# Core dependencies
using Agents
using RxInfer

# Visualization (optional, loaded on demand)
# using GLMakie
# using CairoMakie

# Include type definitions first
include("Brain/Types.jl")
include("World/Types.jl")

# Include core modules
include("Brain/FactorGraph.jl")
include("World/Physics.jl")  # Physics before ActionSelection (ActionSelection uses Physics)
include("Brain/Learning.jl")
include("Brain/ActionSelection.jl")
include("World/Dynamics.jl")

# Include analytics
include("Analytics/Convergence.jl")
include("Analytics/Visualization.jl")

# Re-export key types from submodules
using .BrainTypes
using .WorldTypes
using .FactorGraphs
using .Learning
using .ActionSelection
using .Physics
using .Dynamics
using .Convergence
using .Visualization

# Export public API
export
    # Simulation interface
    Simulation,
    SimulationConfig,
    run_evolution!,
    step_simulation!,

    # Agent types
    InstitutionAgent,
    CognitiveState,
    InteractionRecord,

    # Game types
    GameType,
    PrisonersDilemma,
    StagHunt,
    Harmony,
    resolve_interaction,

    # Active model enum
    ActiveModel,
    NEUTRAL,
    INSTITUTIONAL,

    # Analytics
    institutional_adoption_rate,
    mean_internalization,
    label_correlation,
    compute_free_energy_trajectory,

    # Agents.jl re-exports
    allagents,

    # Visualization
    create_live_dashboard,
    plot_internalization_matrix,
    plot_adoption_trajectory

"""
    SimulationConfig

Configuration parameters for the institution emergence simulation.

# Fields
- `n_agents::Int`: Number of agents (default: 16)
- `game_type::GameType`: Type of social dilemma (default: PrisonersDilemma())
- `complexity_penalty::Float64`: Penalty for adopting institutional model M1
- `initial_precision::Float64`: Starting γ (internalization depth)
- `max_precision::Float64`: Maximum γ value
- `min_precision::Float64`: Minimum γ value
- `γ_update_factor::Float64`: Multiplicative factor for γ updates (default: 1.05)
- `prior_cooperation::Tuple{Float64,Float64}`: Beta prior for cooperation rate
- `action_precision::Float64`: β for action selection softmax
- `structure_learning_threshold::Int`: Min observations before model comparison
- `seed::Union{Int,Nothing}`: Random seed for reproducibility
"""
Base.@kwdef struct SimulationConfig
    n_agents::Int = 16
    game_type::GameType = PrisonersDilemma()
    complexity_penalty::Float64 = 0.1
    initial_precision::Float64 = 1.0
    max_precision::Float64 = 10.0
    min_precision::Float64 = 0.1
    γ_update_factor::Float64 = 1.05
    prior_cooperation::Tuple{Float64,Float64} = (1.0, 1.0)
    action_precision::Float64 = 2.0
    structure_learning_threshold::Int = 10
    seed::Union{Int,Nothing} = nothing
end

"""
    Simulation

Main simulation container holding the Agents.jl model and configuration.
"""
mutable struct Simulation
    model::AgentBasedModel
    config::SimulationConfig
    step_count::Int
    history::DataFrame

    function Simulation(config::SimulationConfig)
        # Set random seed if provided
        if config.seed !== nothing
            Random.seed!(config.seed)
        end

        # Create Agents.jl model
        model = create_institution_model(config)

        # Initialize history DataFrame
        history = DataFrame(
            step = Int[],
            agent_id = Int[],
            label = Bool[],
            active_model = Symbol[],
            precision = Float64[],
            action = Bool[],
            payoff = Float64[]
        )

        new(model, config, 0, history)
    end
end

# Convenience constructor
Simulation(; kwargs...) = Simulation(SimulationConfig(; kwargs...))

"""
    create_institution_model(config::SimulationConfig) -> AgentBasedModel

Create the Agents.jl model with initialized agents.
"""
function create_institution_model(config::SimulationConfig)
    # Model properties (mutable container)
    properties = (
        config = config,
        current_step = Ref(0)
    )

    # Create model without space (agents interact randomly)
    model = StandardABM(
        InstitutionAgent;
        properties = properties,
        scheduler = Schedulers.Randomly()
    )

    # Add agents with random labels
    for i in 1:config.n_agents
        label = rand(Bool)  # Random binary label
        cognitive_state = CognitiveState(
            prior_cooperation = config.prior_cooperation,
            γ = config.initial_precision,
            action_precision = config.action_precision
        )
        # Use add_agent! with positional args for @agent macro
        add_agent!(InstitutionAgent, model, label, cognitive_state, InteractionRecord[])
    end

    return model
end

"""
    run_evolution!(sim::Simulation, steps::Int; verbose=false) -> DataFrame

Run the simulation for a specified number of steps.

Returns a DataFrame with the complete interaction history.
"""
function run_evolution!(sim::Simulation, steps::Int; verbose::Bool=false)
    for _ in 1:steps
        step_simulation!(sim)
        sim.step_count += 1

        if verbose && sim.step_count % 50 == 0
            adoption = institutional_adoption_rate(sim)
            internalization = mean_internalization(sim)
            println("Step $(sim.step_count): adoption=$(round(adoption, digits=2)), γ̄=$(round(internalization, digits=2))")
        end
    end

    return sim.history
end

"""
    step_simulation!(sim::Simulation)

Execute one simulation step: all agents interact pairwise.
"""
function step_simulation!(sim::Simulation)
    model = sim.model
    config = sim.config

    # Get all agents and shuffle for random pairing
    agents = collect(allagents(model))
    shuffle!(agents)

    # Pair agents and execute interactions
    for i in 1:2:length(agents)-1
        agent1 = agents[i]
        agent2 = agents[i+1]

        # Execute interaction
        execute_interaction!(agent1, agent2, config, sim)
    end

    # Handle odd agent (interacts with random partner)
    if isodd(length(agents))
        odd_agent = agents[end]
        partner = rand(agents[1:end-1])
        execute_interaction!(odd_agent, partner, config, sim)
    end

    abmproperties(model).current_step[] += 1
end

"""
    execute_interaction!(agent1, agent2, config, sim)

Execute a single interaction between two agents.
"""
function execute_interaction!(agent1::InstitutionAgent, agent2::InstitutionAgent,
                              config::SimulationConfig, sim::Simulation)
    # Each agent selects action based on opponent's label
    action1 = select_action(agent1, agent2.label, config)
    action2 = select_action(agent2, agent1.label, config)

    # Environment resolves interaction (label-blind)
    payoff1, payoff2 = resolve_interaction(action1, action2, config.game_type)

    # Record observations
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

    push!(agent1.interaction_history, record1)
    push!(agent2.interaction_history, record2)

    # Update beliefs based on observation
    update_beliefs!(agent1, record1, config)
    update_beliefs!(agent2, record2, config)

    # Structure learning check
    if length(agent1.interaction_history) >= config.structure_learning_threshold
        maybe_switch_model!(agent1, config)
    end
    if length(agent2.interaction_history) >= config.structure_learning_threshold
        maybe_switch_model!(agent2, config)
    end

    # Update internalization
    update_internalization!(agent1, record1, config)
    update_internalization!(agent2, record2, config)

    # Record to history
    step = sim.step_count
    push!(sim.history, (step, agent1.id, agent1.label, Symbol(agent1.cognitive_state.active_model),
                        agent1.cognitive_state.γ, action1, payoff1))
    push!(sim.history, (step, agent2.id, agent2.label, Symbol(agent2.cognitive_state.active_model),
                        agent2.cognitive_state.γ, action2, payoff2))
end

end # module

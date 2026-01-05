"""
Default configuration parameters for ArbitraryInstitutions simulations.

This file provides preset configurations for different experimental scenarios.
"""

using ArbitraryInstitutions

# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

"""
Default configuration for standard experiments.
Balanced parameters for observing institution emergence.
"""
const DEFAULT_CONFIG = SimulationConfig(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.1,
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 1.0),  # Uninformative Beta prior
    action_precision = 2.0,
    structure_learning_threshold = 10,
    seed = nothing
)

# ============================================================================
# EXPERIMENTAL PRESETS
# ============================================================================

"""
High complexity penalty - makes institution adoption harder.
Tests whether institutions can still emerge under skeptical agents.
"""
const HIGH_COMPLEXITY_PENALTY = SimulationConfig(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.5,  # Higher penalty
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 1.0),
    action_precision = 2.0,
    structure_learning_threshold = 20,  # Need more evidence
    seed = nothing
)

"""
Low complexity penalty - makes institution adoption easier.
Tests how quickly institutions emerge with credulous agents.
"""
const LOW_COMPLEXITY_PENALTY = SimulationConfig(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.01,  # Lower penalty
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 1.0),
    action_precision = 2.0,
    structure_learning_threshold = 5,  # Less evidence needed
    seed = nothing
)

"""
Stag Hunt game - tests institution role in coordination.
"""
const STAG_HUNT_CONFIG = SimulationConfig(
    n_agents = 16,
    game_type = StagHunt(),
    complexity_penalty = 0.1,
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 1.0),
    action_precision = 2.0,
    structure_learning_threshold = 10,
    seed = nothing
)

"""
Harmony game - control condition where no institution should emerge.
"""
const HARMONY_CONFIG = SimulationConfig(
    n_agents = 16,
    game_type = Harmony(),
    complexity_penalty = 0.1,
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 1.0),
    action_precision = 2.0,
    structure_learning_threshold = 10,
    seed = nothing
)

"""
Large population - tests scalability and emergent dynamics.
"""
const LARGE_POPULATION = SimulationConfig(
    n_agents = 64,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.1,
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 1.0),
    action_precision = 2.0,
    structure_learning_threshold = 10,
    seed = nothing
)

"""
High action precision - more deterministic action selection.
Agents stick more strongly to their EFE-optimal actions.
"""
const DETERMINISTIC_AGENTS = SimulationConfig(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.1,
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 1.0),
    action_precision = 5.0,  # Higher = more deterministic
    structure_learning_threshold = 10,
    seed = nothing
)

"""
Low action precision - more exploratory agents.
"""
const EXPLORATORY_AGENTS = SimulationConfig(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.1,
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 1.0),
    action_precision = 0.5,  # Lower = more random
    structure_learning_threshold = 10,
    seed = nothing
)

"""
Optimistic prior - agents start believing others cooperate.
"""
const OPTIMISTIC_PRIOR = SimulationConfig(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.1,
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (3.0, 1.0),  # Prior mean = 0.75
    action_precision = 2.0,
    structure_learning_threshold = 10,
    seed = nothing
)

"""
Pessimistic prior - agents start believing others defect.
"""
const PESSIMISTIC_PRIOR = SimulationConfig(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.1,
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 3.0),  # Prior mean = 0.25
    action_precision = 2.0,
    structure_learning_threshold = 10,
    seed = nothing
)

# ============================================================================
# REPRODUCIBILITY CONFIGS
# ============================================================================

"""
Reproducible configuration with fixed seed.
Use for debugging and exact replication.
"""
const REPRODUCIBLE = SimulationConfig(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.1,
    initial_precision = 1.0,
    max_precision = 10.0,
    min_precision = 0.1,
    prior_cooperation = (1.0, 1.0),
    action_precision = 2.0,
    structure_learning_threshold = 10,
    seed = 42
)

# ============================================================================
# PARAMETER SWEEP HELPERS
# ============================================================================

"""
Generate configs for parameter sweep over complexity penalty.
"""
function sweep_complexity_penalty(values::Vector{Float64})
    return [
        SimulationConfig(
            n_agents = 16,
            game_type = PrisonersDilemma(),
            complexity_penalty = cp,
            initial_precision = 1.0,
            max_precision = 10.0,
            min_precision = 0.1,
            prior_cooperation = (1.0, 1.0),
            action_precision = 2.0,
            structure_learning_threshold = 10,
            seed = nothing
        )
        for cp in values
    ]
end

"""
Generate configs for parameter sweep over action precision.
"""
function sweep_action_precision(values::Vector{Float64})
    return [
        SimulationConfig(
            n_agents = 16,
            game_type = PrisonersDilemma(),
            complexity_penalty = 0.1,
            initial_precision = 1.0,
            max_precision = 10.0,
            min_precision = 0.1,
            prior_cooperation = (1.0, 1.0),
            action_precision = ap,
            structure_learning_threshold = 10,
            seed = nothing
        )
        for ap in values
    ]
end

"""
Generate configs for all game types.
"""
function all_game_configs()
    return Dict(
        "PrisonersDilemma" => DEFAULT_CONFIG,
        "StagHunt" => STAG_HUNT_CONFIG,
        "Harmony" => HARMONY_CONFIG
    )
end

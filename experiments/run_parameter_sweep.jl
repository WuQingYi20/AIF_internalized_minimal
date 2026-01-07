"""
    Parameter Sweep Framework

Core infrastructure for running systematic parameter exploration experiments.
Provides unified configuration, parallel execution, and result storage.

Usage:
    include("experiments/run_parameter_sweep.jl")
    results = run_experiment(configs, "experiment_name")
"""

using Distributed

# Add workers if not already present
if nworkers() == 1
    addprocs(Sys.CPU_THREADS - 1)
end

@everywhere begin
    using ArbitraryInstitutions
    using Random
    using Statistics: mean, std
    using DataFrames
    using CSV
    using Dates

    # ============================================================
    # Experiment Configuration Types
    # ============================================================

    # Configuration for a single experimental run
    struct ExperimentConfig
        # Base simulation parameters
        n_agents::Int
        game_type::GameType
        complexity_penalty::Float64
        initial_precision::Float64
        max_precision::Float64
        min_precision::Float64
        γ_update_factor::Float64
        prior_cooperation::Tuple{Float64,Float64}
        action_precision::Float64
        structure_learning_threshold::Int

        # Experiment-specific parameters
        bias::Float64              # Initial cooperation bias for True label
        bias_duration::Int         # How long bias is applied
        post_bias_duration::Int    # Steps after bias removal
        seed::Int                  # Random seed for reproducibility

        # Metadata
        experiment_id::String      # Unique experiment identifier
        config_id::Int             # Config index within experiment
    end

    # Results from a single experimental run
    struct ExperimentResult
        # Configuration info
        config::ExperimentConfig

        # Primary outcomes (at end of bias period)
        adoption_at_bias_end::Float64
        gamma_at_bias_end::Float64

        # Final outcomes (after post-bias period)
        final_adoption::Float64
        final_gamma::Float64
        final_correlation::Float64

        # Behavioral outcomes
        true_cooperation_rate::Float64
        false_cooperation_rate::Float64
        cooperation_gap::Float64

        # Belief outcomes
        mean_ingroup_belief::Float64
        mean_outgroup_belief::Float64
        belief_difference::Float64

        # Success indicators
        institution_emerged::Bool       # adoption > 0.5
        self_fulfilling::Bool           # institution_emerged AND coop_gap > 0.05 after bias removal

        # Timing
        elapsed_time::Float64
    end

    # ============================================================
    # Default Configurations
    # ============================================================

    # Create default ExperimentConfig with standard parameters
    function default_config(;
        n_agents::Int = 16,
        game_type::GameType = PrisonersDilemma(),
        complexity_penalty::Float64 = 0.05,
        initial_precision::Float64 = 1.0,
        max_precision::Float64 = 10.0,
        min_precision::Float64 = 0.1,
        γ_update_factor::Float64 = 1.05,
        prior_cooperation::Tuple{Float64,Float64} = (1.0, 1.0),
        action_precision::Float64 = 2.0,
        structure_learning_threshold::Int = 10,
        bias::Float64 = 0.0,
        bias_duration::Int = 100,
        post_bias_duration::Int = 300,
        seed::Int = 1,
        experiment_id::String = "default",
        config_id::Int = 1
    )
        ExperimentConfig(
            n_agents, game_type, complexity_penalty, initial_precision,
            max_precision, min_precision, γ_update_factor, prior_cooperation,
            action_precision, structure_learning_threshold,
            bias, bias_duration, post_bias_duration, seed,
            experiment_id, config_id
        )
    end

    # ============================================================
    # Biased Simulation Functions
    # ============================================================

    # Action selection with optional cooperation bias for True-label agents
    function biased_action_selection(agent, opponent_label, config, bias::Float64)
        game = config.game_type
        G_cooperate = ArbitraryInstitutions.ActionSelection.compute_expected_free_energy(
            agent, true, opponent_label, game
        )
        G_defect = ArbitraryInstitutions.ActionSelection.compute_expected_free_energy(
            agent, false, opponent_label, game
        )

        β = agent.cognitive_state.action_precision
        policy = ArbitraryInstitutions.ActionSelection.softmax([-G_cooperate, -G_defect], β)
        p_cooperate = policy[1]

        # Apply bias only to True-label agents
        if agent.label
            p_cooperate = min(1.0, p_cooperate + bias)
        end

        return rand() < p_cooperate
    end

    # Execute biased interaction between two agents
    function biased_interaction!(agent1, agent2, config, bias::Float64)
        action1 = biased_action_selection(agent1, agent2.label, config, bias)
        action2 = biased_action_selection(agent2, agent1.label, config, bias)

        payoff1, payoff2 = ArbitraryInstitutions.Physics.resolve_interaction(
            action1, action2, config.game_type
        )

        record1 = ArbitraryInstitutions.WorldTypes.InteractionRecord(
            opponent_label = agent2.label,
            opponent_cooperated = action2,
            own_action = action1,
            payoff = payoff1
        )
        record2 = ArbitraryInstitutions.WorldTypes.InteractionRecord(
            opponent_label = agent1.label,
            opponent_cooperated = action1,
            own_action = action2,
            payoff = payoff2
        )

        push!(agent1.interaction_history, record1)
        push!(agent2.interaction_history, record2)

        ArbitraryInstitutions.Learning.update_beliefs!(agent1, record1, config)
        ArbitraryInstitutions.Learning.update_beliefs!(agent2, record2, config)

        if length(agent1.interaction_history) >= config.structure_learning_threshold
            ArbitraryInstitutions.Learning.maybe_switch_model!(agent1, config)
        end
        if length(agent2.interaction_history) >= config.structure_learning_threshold
            ArbitraryInstitutions.Learning.maybe_switch_model!(agent2, config)
        end

        ArbitraryInstitutions.Learning.update_internalization!(agent1, record1, config)
        ArbitraryInstitutions.Learning.update_internalization!(agent2, record2, config)
    end

    # Execute one simulation step with optional bias
    function biased_step!(sim, bias::Float64)
        agents = collect(allagents(sim.model))
        shuffle!(agents)

        for i in 1:2:length(agents)-1
            biased_interaction!(agents[i], agents[i+1], sim.config, bias)
        end

        if isodd(length(agents))
            biased_interaction!(agents[end], rand(agents[1:end-1]), sim.config, bias)
        end

        sim.step_count += 1
    end

    # ============================================================
    # Single Experiment Run
    # ============================================================

    # Run a single experiment configuration and return results
    function run_single_experiment(exp_config::ExperimentConfig)::ExperimentResult
        start_time = time()

        # Create simulation with experiment parameters
        sim_config = SimulationConfig(
            n_agents = exp_config.n_agents,
            game_type = exp_config.game_type,
            complexity_penalty = exp_config.complexity_penalty,
            initial_precision = exp_config.initial_precision,
            max_precision = exp_config.max_precision,
            min_precision = exp_config.min_precision,
            γ_update_factor = exp_config.γ_update_factor,
            prior_cooperation = exp_config.prior_cooperation,
            action_precision = exp_config.action_precision,
            structure_learning_threshold = exp_config.structure_learning_threshold,
            seed = exp_config.seed
        )

        Random.seed!(exp_config.seed)
        sim = Simulation(sim_config)

        # Phase 1: Biased period
        for _ in 1:exp_config.bias_duration
            biased_step!(sim, exp_config.bias)
        end

        adoption_at_bias_end = institutional_adoption_rate(sim)
        gamma_at_bias_end = mean_internalization(sim)

        # Phase 2: Post-bias period (no bias)
        for _ in 1:exp_config.post_bias_duration
            biased_step!(sim, 0.0)
        end

        # Compute final metrics
        final_adoption = institutional_adoption_rate(sim)
        final_gamma = mean_internalization(sim)
        final_correlation = label_correlation(sim)

        # Compute cooperation rates by label
        true_agents = filter(a -> a.label, collect(allagents(sim.model)))
        false_agents = filter(a -> !a.label, collect(allagents(sim.model)))

        true_coop_rate = compute_recent_cooperation(true_agents, 50)
        false_coop_rate = compute_recent_cooperation(false_agents, 50)
        cooperation_gap = true_coop_rate - false_coop_rate

        # Compute belief differences
        mean_ingroup, mean_outgroup = compute_mean_beliefs(collect(allagents(sim.model)))
        belief_difference = mean_ingroup - mean_outgroup

        # Success indicators
        institution_emerged = final_adoption > 0.5
        self_fulfilling = institution_emerged && abs(cooperation_gap) > 0.05

        elapsed_time = time() - start_time

        return ExperimentResult(
            exp_config,
            adoption_at_bias_end, gamma_at_bias_end,
            final_adoption, final_gamma, final_correlation,
            true_coop_rate, false_coop_rate, cooperation_gap,
            mean_ingroup, mean_outgroup, belief_difference,
            institution_emerged, self_fulfilling,
            elapsed_time
        )
    end

    # Compute recent cooperation rate for a group of agents
    function compute_recent_cooperation(agents, window::Int)
        total_coop = 0
        total_interactions = 0

        for agent in agents
            history = agent.interaction_history
            recent = history[max(1, end-window+1):end]
            for r in recent
                total_interactions += 1
                if r.own_action
                    total_coop += 1
                end
            end
        end

        return total_interactions > 0 ? total_coop / total_interactions : 0.0
    end

    # Compute mean ingroup and outgroup beliefs across all agents
    function compute_mean_beliefs(agents)
        ingroup_beliefs = Float64[]
        outgroup_beliefs = Float64[]

        for agent in agents
            beliefs = agent.cognitive_state.beliefs
            push!(ingroup_beliefs, beliefs.α_ingroup / (beliefs.α_ingroup + beliefs.β_ingroup))
            push!(outgroup_beliefs, beliefs.α_outgroup / (beliefs.α_outgroup + beliefs.β_outgroup))
        end

        return mean(ingroup_beliefs), mean(outgroup_beliefs)
    end

end  # @everywhere

# ============================================================
# Batch Experiment Runner (main process only)
# ============================================================

"""
Run multiple experiment configurations in parallel.
Returns DataFrame with all results.
"""
function run_experiment_batch(configs::Vector{ExperimentConfig};
                              show_progress::Bool = true)::DataFrame
    n_configs = length(configs)

    if show_progress
        println("Running $(n_configs) configurations on $(nworkers()) workers...")
    end

    # Run in parallel
    results = pmap(run_single_experiment, configs)

    # Convert to DataFrame
    df = results_to_dataframe(results)

    if show_progress
        println("Completed $(n_configs) runs.")
    end

    return df
end

"""
Convert vector of ExperimentResult to DataFrame.
"""
function results_to_dataframe(results::Vector{ExperimentResult})::DataFrame
    df = DataFrame(
        # Config info
        experiment_id = [r.config.experiment_id for r in results],
        config_id = [r.config.config_id for r in results],
        seed = [r.config.seed for r in results],

        # Simulation parameters
        n_agents = [r.config.n_agents for r in results],
        game_type = [string(typeof(r.config.game_type)) for r in results],
        complexity_penalty = [r.config.complexity_penalty for r in results],
        initial_precision = [r.config.initial_precision for r in results],
        max_precision = [r.config.max_precision for r in results],
        min_precision = [r.config.min_precision for r in results],
        γ_update_factor = [r.config.γ_update_factor for r in results],
        prior_α = [r.config.prior_cooperation[1] for r in results],
        prior_β = [r.config.prior_cooperation[2] for r in results],
        action_precision = [r.config.action_precision for r in results],
        structure_learning_threshold = [r.config.structure_learning_threshold for r in results],

        # Experiment parameters
        bias = [r.config.bias for r in results],
        bias_duration = [r.config.bias_duration for r in results],
        post_bias_duration = [r.config.post_bias_duration for r in results],

        # Results at bias end
        adoption_at_bias_end = [r.adoption_at_bias_end for r in results],
        gamma_at_bias_end = [r.gamma_at_bias_end for r in results],

        # Final results
        final_adoption = [r.final_adoption for r in results],
        final_gamma = [r.final_gamma for r in results],
        final_correlation = [r.final_correlation for r in results],

        # Behavioral outcomes
        true_cooperation_rate = [r.true_cooperation_rate for r in results],
        false_cooperation_rate = [r.false_cooperation_rate for r in results],
        cooperation_gap = [r.cooperation_gap for r in results],

        # Belief outcomes
        mean_ingroup_belief = [r.mean_ingroup_belief for r in results],
        mean_outgroup_belief = [r.mean_outgroup_belief for r in results],
        belief_difference = [r.belief_difference for r in results],

        # Success indicators
        institution_emerged = [r.institution_emerged for r in results],
        self_fulfilling = [r.self_fulfilling for r in results],

        # Performance
        elapsed_time = [r.elapsed_time for r in results]
    )

    return df
end

# ============================================================
# Configuration Generators
# ============================================================

"""
Generate configurations for parameter sweep with multiple seeds.
"""
function generate_sweep_configs(;
    experiment_id::String,
    param_grid::Dict,
    n_repeats::Int = 20,
    base_seed::Int = 1000
)::Vector{ExperimentConfig}

    configs = ExperimentConfig[]
    config_id = 1

    # Get all parameter combinations
    param_names = collect(keys(param_grid))
    param_values = collect(values(param_grid))

    for combo in Iterators.product(param_values...)
        params = Dict(zip(param_names, combo))

        for rep in 1:n_repeats
            seed = base_seed + config_id * 100 + rep

            config = default_config(;
                experiment_id = experiment_id,
                config_id = config_id,
                seed = seed,
                params...
            )

            push!(configs, config)
            config_id += 1
        end
    end

    return configs
end

# ============================================================
# Result Storage
# ============================================================

"""
Save experiment results to CSV with metadata.
"""
function save_results(df::DataFrame, experiment_id::String;
                      output_dir::String = "experiments/results")

    # Create output directory if needed
    mkpath(output_dir)

    # Generate filename with timestamp
    timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
    filename = joinpath(output_dir, "$(experiment_id)_$(timestamp).csv")

    # Save CSV
    CSV.write(filename, df)

    println("Results saved to: $filename")

    return filename
end

"""
Load and combine multiple result files for an experiment.
"""
function load_results(experiment_id::String;
                      results_dir::String = "experiments/results")::DataFrame

    # Find all matching files
    files = filter(f -> startswith(f, experiment_id) && endswith(f, ".csv"),
                   readdir(results_dir))

    if isempty(files)
        error("No results found for experiment: $experiment_id")
    end

    # Load and combine
    dfs = [CSV.read(joinpath(results_dir, f), DataFrame) for f in files]

    return vcat(dfs...)
end

# ============================================================
# Summary Statistics
# ============================================================

"""
Compute summary statistics grouped by parameter combinations.
"""
function summarize_results(df::DataFrame, group_cols::Vector{Symbol})::DataFrame
    gdf = groupby(df, group_cols)

    summary = combine(gdf,
        :final_adoption => mean => :mean_adoption,
        :final_adoption => std => :std_adoption,
        :final_gamma => mean => :mean_gamma,
        :final_gamma => std => :std_gamma,
        :cooperation_gap => mean => :mean_coop_gap,
        :cooperation_gap => std => :std_coop_gap,
        :institution_emerged => mean => :emergence_rate,
        :self_fulfilling => mean => :self_fulfilling_rate,
        :elapsed_time => mean => :mean_time,
        nrow => :n_runs
    )

    return summary
end

# ============================================================
# Quick Test Function
# ============================================================

"""
Run a quick test to verify the framework works.
"""
function test_framework()
    println("Testing experiment framework...")

    # Create a few test configs
    configs = [
        default_config(bias=0.0, seed=1, experiment_id="test", config_id=1),
        default_config(bias=0.1, seed=2, experiment_id="test", config_id=2),
        default_config(bias=0.2, seed=3, experiment_id="test", config_id=3)
    ]

    # Run (using single worker for test)
    results = map(run_single_experiment, configs)

    println("\nTest Results:")
    for r in results
        println("  bias=$(r.config.bias): adoption=$(round(r.final_adoption, digits=2)), " *
                "gamma=$(round(r.final_gamma, digits=2)), gap=$(round(r.cooperation_gap*100, digits=1))%")
    end

    println("\nFramework test passed!")
end

println("Parameter sweep framework loaded.")
println("Use test_framework() to verify setup.")
println("Use run_experiment_batch(configs) to run experiments.")

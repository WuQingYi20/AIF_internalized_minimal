"""
    Experiment 5: Population Scale Effect

Goal: Understand critical mass and scale relationships.

Parameter grid:
- n_agents: [8, 16, 32, 64, 128]
- bias: [0.05, 0.10, 0.15]

Total configurations: 5 × 3 = 15
With 20 repeats: 300 runs

Predictions:
- Small population → more noise → harder to emerge
- Large population → more stable signal → easier to emerge?
  Or harder to reach critical mass?
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")

const EXPERIMENT_ID = "exp5_population_scale"
const N_REPEATS = 20

function run_experiment5()
    println("=" ^ 80)
    println("Experiment 5: Population Scale Effect")
    println("=" ^ 80)

    # Define parameter grid
    agent_counts = [8, 16, 32, 64, 128]
    biases = [0.05, 0.10, 0.15]

    configs = ExperimentConfig[]
    config_id = 1

    for n in agent_counts
        for bias in biases
            for rep in 1:N_REPEATS
                seed = 5000 + config_id * 100 + rep

                config = default_config(
                    experiment_id = EXPERIMENT_ID,
                    config_id = config_id,
                    seed = seed,
                    n_agents = n,
                    bias = bias,
                    bias_duration = 100,
                    post_bias_duration = 300
                )
                push!(configs, config)
            end
            config_id += 1
        end
    end

    println("Total configurations: $(length(configs))")
    println("Running on $(nworkers()) workers...")

    # Run experiments
    df = run_experiment_batch(configs)

    # Save results
    save_results(df, EXPERIMENT_ID)

    # Print summary
    println("\n" * "=" ^ 80)
    println("Summary by Population Size and Bias")
    println("=" ^ 80)

    summary = summarize_results(df, [:n_agents, :bias])
    sort!(summary, [:n_agents, :bias])

    println("\nN Agents | Bias | Adoption | Gamma | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 75)

    for row in eachrow(summary)
        @printf("   %3d   | %.2f |  %5.1f%% | %.2f  |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            row.n_agents,
            row.bias,
            row.mean_adoption * 100,
            row.mean_gamma,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Scale analysis
    println("\n" * "=" ^ 80)
    println("Scale Effect Analysis")
    println("=" ^ 80)

    summary_n = summarize_results(df, [:n_agents])
    sort!(summary_n, :n_agents)

    println("\nPopulation effect (averaged over biases):")
    println("\nN Agents | Emergence | Self-Fulfill | Mean Time (s)")
    println("-" ^ 50)

    for row in eachrow(summary_n)
        @printf("   %3d   |   %5.1f%%  |    %5.1f%%   |    %.2f\n",
            row.n_agents,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_time
        )
    end

    # Check for threshold effects
    println("\n" * "=" ^ 80)
    println("Critical Mass Analysis")
    println("=" ^ 80)

    for bias in biases
        bias_data = filter(row -> row.bias == bias, summary)

        println("\nBias = $bias:")
        for row in eachrow(bias_data)
            n_adopters = round(row.mean_adoption * row.n_agents, digits=1)
            println("  N=$(row.n_agents): $(round(row.mean_adoption*100, digits=1))% adoption " *
                    "≈ $(n_adopters) agents")
        end
    end

    return df
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment5()
end

"""
    Experiment 3: Action Precision Effect

Goal: Explore how exploration-exploitation trade-off affects emergence.

Parameter grid:
- action_precision (β): [0.5, 1.0, 2.0, 5.0, 10.0]
- bias: [0.05, 0.10, 0.15]

Total configurations: 5 × 3 = 15
With 20 repeats: 300 runs

Predictions:
- Low β → more exploration → more noise → harder to detect signal
- High β → more greedy → clearer signal but may lock in early
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")

const EXPERIMENT_ID = "exp3_action_precision"
const N_REPEATS = 20

function run_experiment3()
    println("=" ^ 80)
    println("Experiment 3: Action Precision Effect")
    println("=" ^ 80)

    # Define parameter grid
    precisions = [0.5, 1.0, 2.0, 5.0, 10.0]
    biases = [0.05, 0.10, 0.15]

    configs = ExperimentConfig[]
    config_id = 1

    for precision in precisions
        for bias in biases
            for rep in 1:N_REPEATS
                seed = 3000 + config_id * 100 + rep

                config = default_config(
                    experiment_id = EXPERIMENT_ID,
                    config_id = config_id,
                    seed = seed,
                    action_precision = precision,
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
    println("Summary by Action Precision and Bias")
    println("=" ^ 80)

    summary = summarize_results(df, [:action_precision, :bias])
    sort!(summary, [:action_precision, :bias])

    println("\nβ    | Bias | Adoption | Gamma | Emergence | Self-Fulfill | Coop Gap | Belief Diff")
    println("-" ^ 85)

    # Need to compute belief difference in summary
    gdf = groupby(df, [:action_precision, :bias])
    summary_full = combine(gdf,
        :final_adoption => mean => :mean_adoption,
        :final_gamma => mean => :mean_gamma,
        :cooperation_gap => mean => :mean_coop_gap,
        :belief_difference => mean => :mean_belief_diff,
        :institution_emerged => mean => :emergence_rate,
        :self_fulfilling => mean => :self_fulfilling_rate
    )
    sort!(summary_full, [:action_precision, :bias])

    for row in eachrow(summary_full)
        @printf("%.1f  | %.2f |  %5.1f%% | %.2f  |   %5.1f%%  |    %5.1f%%   | %+5.1f%% | %+5.2f\n",
            row.action_precision,
            row.bias,
            row.mean_adoption * 100,
            row.mean_gamma,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_coop_gap * 100,
            row.mean_belief_diff
        )
    end

    # Analyze optimal precision
    println("\n" * "=" ^ 80)
    println("Exploration-Exploitation Trade-off Analysis")
    println("=" ^ 80)

    for bias in biases
        bias_data = filter(row -> row.bias == bias, summary_full)
        best_idx = argmax(bias_data.self_fulfilling_rate)

        println("\nBias = $bias:")
        println("  Best action precision: $(bias_data.action_precision[best_idx])")
        println("  Self-fulfilling rate: $(round(bias_data.self_fulfilling_rate[best_idx] * 100, digits=1))%")
        println("  Cooperation gap: $(round(bias_data.mean_coop_gap[best_idx] * 100, digits=1))%")
    end

    return df
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment3()
end

"""
    Experiment 2: Complexity Penalty Sensitivity

Goal: Understand how Occam's razor strength affects institution emergence.

Parameter grid:
- complexity_penalty: [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
- bias: [0.05, 0.10, 0.15]

Total configurations: 7 × 3 = 21
With 20 repeats: 420 runs

Predictions:
- Low penalty → easier to adopt M1 → more institution emergence
- High penalty → need stronger signal to switch
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")

const EXPERIMENT_ID = "exp2_complexity_penalty"
const N_REPEATS = 20

function run_experiment2()
    println("=" ^ 80)
    println("Experiment 2: Complexity Penalty Sensitivity")
    println("=" ^ 80)

    # Define parameter grid
    penalties = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    biases = [0.05, 0.10, 0.15]

    configs = ExperimentConfig[]
    config_id = 1

    for penalty in penalties
        for bias in biases
            for rep in 1:N_REPEATS
                seed = 2000 + config_id * 100 + rep

                config = default_config(
                    experiment_id = EXPERIMENT_ID,
                    config_id = config_id,
                    seed = seed,
                    complexity_penalty = penalty,
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
    println("Summary by Complexity Penalty and Bias")
    println("=" ^ 80)

    summary = summarize_results(df, [:complexity_penalty, :bias])
    sort!(summary, [:complexity_penalty, :bias])

    println("\nPenalty | Bias | Adoption | Gamma | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 75)

    for row in eachrow(summary)
        @printf("%.3f   | %.2f |  %5.1f%% | %.2f  |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            row.complexity_penalty,
            row.bias,
            row.mean_adoption * 100,
            row.mean_gamma,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Analyze threshold effect
    println("\n" * "=" ^ 80)
    println("Optimal Complexity Penalty Analysis")
    println("=" ^ 80)

    for bias in biases
        bias_data = filter(row -> row.bias == bias, summary)
        best_row = argmax(bias_data.self_fulfilling_rate)

        println("\nBias = $bias:")
        println("  Best penalty: $(bias_data.complexity_penalty[best_row])")
        println("  Self-fulfilling rate: $(round(bias_data.self_fulfilling_rate[best_row] * 100, digits=1))%")
    end

    return df
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment2()
end

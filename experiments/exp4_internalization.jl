"""
    Experiment 4: Internalization Dynamics

Goal: Understand how γ update parameters affect the feedback loop.

Parameter grid:
- γ_update_factor: [1.01, 1.05, 1.10, 1.20]
- initial_precision: [0.5, 1.0, 2.0]
- max_precision: [5.0, 10.0, 20.0]

Total configurations: 4 × 3 × 3 = 36
With 20 repeats: 720 runs

Key questions:
- Does faster γ update accelerate emergence?
- Or does it lead to overfitting?
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")

const EXPERIMENT_ID = "exp4_internalization"
const N_REPEATS = 20

function run_experiment4()
    println("=" ^ 80)
    println("Experiment 4: Internalization Dynamics")
    println("=" ^ 80)

    # Define parameter grid
    γ_factors = [1.01, 1.05, 1.10, 1.20]
    initial_precs = [0.5, 1.0, 2.0]
    max_precs = [5.0, 10.0, 20.0]

    # Fixed bias at moderate level
    bias = 0.10

    configs = ExperimentConfig[]
    config_id = 1

    for γ_factor in γ_factors
        for init_prec in initial_precs
            for max_prec in max_precs
                for rep in 1:N_REPEATS
                    seed = 4000 + config_id * 100 + rep

                    config = default_config(
                        experiment_id = EXPERIMENT_ID,
                        config_id = config_id,
                        seed = seed,
                        γ_update_factor = γ_factor,
                        initial_precision = init_prec,
                        max_precision = max_prec,
                        bias = bias,
                        bias_duration = 100,
                        post_bias_duration = 300
                    )
                    push!(configs, config)
                end
                config_id += 1
            end
        end
    end

    println("Total configurations: $(length(configs))")
    println("Running on $(nworkers()) workers...")

    # Run experiments
    df = run_experiment_batch(configs)

    # Save results
    save_results(df, EXPERIMENT_ID)

    # Print summary by γ_update_factor
    println("\n" * "=" ^ 80)
    println("Summary by γ Update Factor")
    println("=" ^ 80)

    summary_γ = summarize_results(df, [:γ_update_factor])
    sort!(summary_γ, :γ_update_factor)

    println("\nγ_factor | Adoption | Final γ | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 70)

    for row in eachrow(summary_γ)
        @printf("  %.2f   |  %5.1f%% |  %.2f   |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            row.γ_update_factor,
            row.mean_adoption * 100,
            row.mean_gamma,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Summary by initial precision
    println("\n" * "=" ^ 80)
    println("Summary by Initial Precision")
    println("=" ^ 80)

    summary_init = summarize_results(df, [:initial_precision])
    sort!(summary_init, :initial_precision)

    println("\nInit γ | Adoption | Final γ | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 70)

    for row in eachrow(summary_init)
        @printf("  %.1f  |  %5.1f%% |  %.2f   |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            row.initial_precision,
            row.mean_adoption * 100,
            row.mean_gamma,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Summary by max precision
    println("\n" * "=" ^ 80)
    println("Summary by Max Precision")
    println("=" ^ 80)

    summary_max = summarize_results(df, [:max_precision])
    sort!(summary_max, :max_precision)

    println("\nMax γ | Adoption | Final γ | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 70)

    for row in eachrow(summary_max)
        @printf("  %.1f |  %5.1f%% |  %.2f   |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            row.max_precision,
            row.mean_adoption * 100,
            row.mean_gamma,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Full interaction analysis
    println("\n" * "=" ^ 80)
    println("Full Parameter Interaction")
    println("=" ^ 80)

    summary_full = summarize_results(df, [:γ_update_factor, :initial_precision, :max_precision])
    best_config = summary_full[argmax(summary_full.self_fulfilling_rate), :]

    println("\nBest configuration:")
    println("  γ_update_factor: $(best_config.γ_update_factor)")
    println("  initial_precision: $(best_config.initial_precision)")
    println("  max_precision: $(best_config.max_precision)")
    println("  Self-fulfilling rate: $(round(best_config.self_fulfilling_rate * 100, digits=1))%")

    return df
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment4()
end

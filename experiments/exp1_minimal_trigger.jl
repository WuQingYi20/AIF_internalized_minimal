"""
    Experiment 1: Minimal Trigger Conditions

Goal: Find the minimum bias needed to trigger self-fulfilling prophecy.

Parameter grid:
- bias: [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
- bias_duration: [25, 50, 100, 200]
- game_type: [PrisonersDilemma, StagHunt]

Total configurations: 6 × 4 × 2 = 48
With 20 repeats: 960 runs
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")
using Printf

const EXPERIMENT_ID = "exp1_minimal_trigger"
const N_REPEATS = 20

function run_experiment1()
    println("=" ^ 80)
    println("Experiment 1: Minimal Trigger Conditions")
    println("=" ^ 80)

    # Define parameter grid
    biases = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    durations = [25, 50, 100, 200]
    game_types = [PrisonersDilemma(), StagHunt()]

    configs = ExperimentConfig[]
    config_id = 1

    for game in game_types
        for duration in durations
            for bias in biases
                for rep in 1:N_REPEATS
                    seed = 1000 + config_id * 100 + rep

                    config = default_config(
                        experiment_id = EXPERIMENT_ID,
                        config_id = config_id,
                        seed = seed,
                        game_type = game,
                        bias = bias,
                        bias_duration = duration,
                        post_bias_duration = 300  # Standard post-bias observation
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

    # Print summary
    println("\n" * "=" ^ 80)
    println("Summary by Game Type and Bias")
    println("=" ^ 80)

    summary = summarize_results(df, [:game_type, :bias, :bias_duration])
    sort!(summary, [:game_type, :bias_duration, :bias])

    println("\nGame Type      | Duration | Bias | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 75)

    for row in eachrow(summary)
        game_short = contains(row.game_type, "Prisoners") ? "PD" : "SH"
        @printf("%14s | %4d     | %.2f |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            game_short,
            row.bias_duration,
            row.bias,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Identify critical threshold
    println("\n" * "=" ^ 80)
    println("Critical Threshold Analysis")
    println("=" ^ 80)

    for game in ["PrisonersDilemma", "StagHunt"]
        game_data = filter(row -> contains(row.game_type, game), summary)
        critical = filter(row -> row.self_fulfilling_rate > 0.5, game_data)

        if !isempty(critical)
            min_bias = minimum(critical.bias)
            min_duration = minimum(filter(row -> row.bias == min_bias, critical).bias_duration)
            println("\n$(game):")
            println("  Minimum bias for >50% self-fulfilling: $(min_bias)")
            println("  Minimum duration at that bias: $(min_duration) steps")
        else
            println("\n$(game): No configuration achieved >50% self-fulfilling rate")
        end
    end

    return df
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment1()
end

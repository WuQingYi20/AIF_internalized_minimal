"""
    Experiment 6: Game Type Comparison

Goal: Understand how different social dilemmas affect institution emergence.

Parameter grid:
- game_type: [PrisonersDilemma, StagHunt, Harmony]
- bias: [0.05, 0.10, 0.15, 0.20]
- complexity_penalty: [0.01, 0.05, 0.1]

Total configurations: 3 × 4 × 3 = 36
With 20 repeats: 720 runs

Theoretical predictions:
- PD: Needs reciprocity mechanism, hardest for institutions
- StagHunt: Coordination game, institutions should emerge easier
- Harmony: No dilemma, institutions "unnecessary"
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")
using Printf

const EXPERIMENT_ID = "exp6_game_types"
const N_REPEATS = 20

function run_experiment6()
    println("=" ^ 80)
    println("Experiment 6: Game Type Comparison")
    println("=" ^ 80)

    # Define parameter grid
    game_types = [PrisonersDilemma(), StagHunt(), Harmony()]
    biases = [0.05, 0.10, 0.15, 0.20]
    penalties = [0.01, 0.05, 0.1]

    configs = ExperimentConfig[]
    config_id = 1

    for game in game_types
        for bias in biases
            for penalty in penalties
                for rep in 1:N_REPEATS
                    seed = 6000 + config_id * 100 + rep

                    config = default_config(
                        experiment_id = EXPERIMENT_ID,
                        config_id = config_id,
                        seed = seed,
                        game_type = game,
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
    end

    println("Total configurations: $(length(configs))")
    println("Running on $(nworkers()) workers...")

    # Run experiments
    df = run_experiment_batch(configs)

    # Save results
    save_results(df, EXPERIMENT_ID)

    # Summary by game type
    println("\n" * "=" ^ 80)
    println("Summary by Game Type")
    println("=" ^ 80)

    summary_game = summarize_results(df, [:game_type])

    println("\nGame Type         | Adoption | Gamma | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 80)

    for row in eachrow(summary_game)
        game_name = replace(row.game_type, "()" => "")
        @printf("%-17s |  %5.1f%% | %.2f  |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            game_name,
            row.mean_adoption * 100,
            row.mean_gamma,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Full breakdown
    println("\n" * "=" ^ 80)
    println("Summary by Game Type and Bias")
    println("=" ^ 80)

    summary_full = summarize_results(df, [:game_type, :bias])
    sort!(summary_full, [:game_type, :bias])

    println("\nGame Type         | Bias | Adoption | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 80)

    for row in eachrow(summary_full)
        game_name = replace(row.game_type, "()" => "")
        @printf("%-17s | %.2f |  %5.1f%% |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            game_name,
            row.bias,
            row.mean_adoption * 100,
            row.emergence_rate * 100,
            row.self_fulfilling_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Theoretical validation
    println("\n" * "=" ^ 80)
    println("Theoretical Validation")
    println("=" ^ 80)

    println("""
    Expected Results:
    1. Prisoner's Dilemma:
       - Hardest for institution emergence
       - Requires reciprocity mechanism to escape mutual defection
       - Self-fulfilling prophecy enables cooperation clusters

    2. Stag Hunt:
       - Easier than PD (coordination game)
       - Once enough agents believe in cooperation, it's mutually optimal
       - Should see faster institution emergence

    3. Harmony:
       - No social dilemma
       - Cooperation always optimal regardless of beliefs
       - Institution adoption may occur but is "unnecessary"
    """)

    # Check if results match theory
    summary_by_game = combine(groupby(df, :game_type),
        :self_fulfilling => mean => :sf_rate,
        :cooperation_gap => mean => :gap
    )

    println("\nEmpirical Results:")
    for row in eachrow(summary_by_game)
        game_short = contains(row.game_type, "Prisoners") ? "PD" :
                     contains(row.game_type, "Stag") ? "SH" : "H"
        println("  $game_short: Self-fulfilling=$(round(row.sf_rate*100, digits=1))%, " *
                "Coop Gap=$(round(row.gap*100, digits=1))%")
    end

    return df
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment6()
end

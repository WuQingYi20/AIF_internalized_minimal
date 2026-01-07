"""
    Experiment 8: Long-term Stability

Goal: Test whether institutions persist indefinitely or eventually collapse.

Parameter grid:
- simulation_length (total steps): [500, 1000, 2000, 5000]
- bias: [0.10, 0.15, 0.20]
- bias_duration: [100, 200]

Total configurations: 4 × 3 × 2 = 24
With 20 repeats: 480 runs

Key questions:
- Do institutions remain stable in the long run?
- Is there "institutional decay" over time?
- Does the Bayesian learning eventually eliminate false beliefs?
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")
using Printf

const EXPERIMENT_ID = "exp8_long_term"
const N_REPEATS = 20

function run_experiment8()
    println("=" ^ 80)
    println("Experiment 8: Long-term Stability")
    println("=" ^ 80)

    # Define parameter grid
    total_lengths = [500, 1000, 2000, 5000]
    biases = [0.10, 0.15, 0.20]
    bias_durations = [100, 200]

    configs = ExperimentConfig[]
    config_id = 1

    for total_len in total_lengths
        for bias_dur in bias_durations
            post_dur = total_len - bias_dur
            if post_dur <= 0
                continue
            end

            for bias in biases
                for rep in 1:N_REPEATS
                    seed = 8000 + config_id * 100 + rep

                    config = default_config(
                        experiment_id = EXPERIMENT_ID,
                        config_id = config_id,
                        seed = seed,
                        bias = bias,
                        bias_duration = bias_dur,
                        post_bias_duration = post_dur
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

    # Add total length column
    df.total_length = df.bias_duration .+ df.post_bias_duration

    # Summary by total length
    println("\n" * "=" ^ 80)
    println("Summary by Simulation Length")
    println("=" ^ 80)

    gdf = groupby(df, :total_length)
    summary_len = combine(gdf,
        :final_adoption => mean => :mean_adoption,
        :final_gamma => mean => :mean_gamma,
        :cooperation_gap => mean => :mean_coop_gap,
        :institution_emerged => mean => :emergence_rate,
        :self_fulfilling => mean => :sf_rate,
        :belief_difference => mean => :mean_belief_diff
    )
    sort!(summary_len, :total_length)

    println("\nLength | Adoption | Gamma | Emergence | Self-Fulfill | Coop Gap | Belief Diff")
    println("-" ^ 80)

    for row in eachrow(summary_len)
        @printf("%5d  |  %5.1f%% | %.2f  |   %5.1f%%  |    %5.1f%%   | %+5.1f%% |   %+.3f\n",
            row.total_length,
            row.mean_adoption * 100,
            row.mean_gamma,
            row.emergence_rate * 100,
            row.sf_rate * 100,
            row.mean_coop_gap * 100,
            row.mean_belief_diff
        )
    end

    # Detailed breakdown
    println("\n" * "=" ^ 80)
    println("Summary by Length and Bias")
    println("=" ^ 80)

    gdf2 = groupby(df, [:total_length, :bias])
    summary_full = combine(gdf2,
        :final_adoption => mean => :mean_adoption,
        :cooperation_gap => mean => :mean_coop_gap,
        :self_fulfilling => mean => :sf_rate
    )
    sort!(summary_full, [:bias, :total_length])

    println("\nBias | Length | Adoption | Self-Fulfill | Coop Gap")
    println("-" ^ 55)

    for row in eachrow(summary_full)
        @printf("%.2f | %5d  |  %5.1f%% |    %5.1f%%   | %+5.1f%%\n",
            row.bias,
            row.total_length,
            row.mean_adoption * 100,
            row.sf_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Stability analysis
    println("\n" * "=" ^ 80)
    println("Institutional Stability Analysis")
    println("=" ^ 80)

    # Compare early vs late
    for bias in biases
        bias_data = filter(row -> row.bias == bias, summary_full)
        if nrow(bias_data) >= 2
            early = bias_data[1, :]  # Shortest run
            late = bias_data[end, :]  # Longest run

            println("\nBias = $bias:")
            println("  Short run ($(early.total_length) steps):")
            println("    Self-fulfilling: $(round(early.sf_rate * 100, digits=1))%")
            println("    Coop gap: $(round(early.mean_coop_gap * 100, digits=1))%")
            println("  Long run ($(late.total_length) steps):")
            println("    Self-fulfilling: $(round(late.sf_rate * 100, digits=1))%")
            println("    Coop gap: $(round(late.mean_coop_gap * 100, digits=1))%")

            # Check for decay
            if late.sf_rate < early.sf_rate - 0.1
                println("  ⚠ INSTITUTIONAL DECAY DETECTED")
            elseif late.sf_rate > early.sf_rate - 0.05
                println("  ✓ Institution remains stable")
            end
        end
    end

    # Theoretical discussion
    println("\n" * "=" ^ 80)
    println("Theoretical Discussion")
    println("=" ^ 80)

    println("""
    Long-term Stability Mechanisms:

    1. Self-Reinforcing Loop:
       Belief → Action → Observation → Belief
       If this loop is strong, institutions persist indefinitely.

    2. Bayesian Truth-Seeking:
       Over time, beliefs should converge to true cooperation rates.
       In label-blind environment, true rates are EQUAL for both groups.
       This should eventually eliminate institutional beliefs.

    3. Equilibrium Question:
       Can the behavioral difference generated by beliefs be large enough
       to sustain the beliefs that generate it?

    Key Finding:
       If self-fulfilling rate remains HIGH at 5000 steps,
       institutions have achieved stable equilibrium.
       If it decays, Bayesian learning eventually overcomes the prophecy.
    """)

    return df
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment8()
end

"""
    Experiment 7: Prior Belief Effect

Goal: Understand how initial optimism/pessimism affects institution emergence.

Prior settings:
- Pessimistic: (1, 3) → initial expected cooperation 25%
- Neutral: (1, 1) → initial expected cooperation 50%
- Optimistic: (3, 1) → initial expected cooperation 75%

Parameter grid:
- prior_cooperation: [(1,3), (1,1), (3,1)]
- game_type: [PrisonersDilemma, StagHunt]
- bias: [0.05, 0.10, 0.15]

Total configurations: 3 × 2 × 3 = 18
With 20 repeats: 360 runs
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")
using Printf

const EXPERIMENT_ID = "exp7_prior_beliefs"
const N_REPEATS = 20

function run_experiment7()
    println("=" ^ 80)
    println("Experiment 7: Prior Belief Effect")
    println("=" ^ 80)

    # Define parameter grid
    priors = [
        (1.0, 3.0),  # Pessimistic: E[p] = 0.25
        (1.0, 1.0),  # Neutral: E[p] = 0.50
        (3.0, 1.0)   # Optimistic: E[p] = 0.75
    ]
    prior_names = ["Pessimistic", "Neutral", "Optimistic"]

    game_types = [PrisonersDilemma(), StagHunt()]
    biases = [0.05, 0.10, 0.15]

    configs = ExperimentConfig[]
    config_id = 1

    for (prior, pname) in zip(priors, prior_names)
        for game in game_types
            for bias in biases
                for rep in 1:N_REPEATS
                    seed = 7000 + config_id * 100 + rep

                    config = default_config(
                        experiment_id = EXPERIMENT_ID,
                        config_id = config_id,
                        seed = seed,
                        prior_cooperation = prior,
                        game_type = game,
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

    # Add prior name column for easier analysis
    df.prior_name = map(row -> begin
        α, β = row.prior_α, row.prior_β
        if α == 1.0 && β == 3.0
            "Pessimistic"
        elseif α == 1.0 && β == 1.0
            "Neutral"
        else
            "Optimistic"
        end
    end, eachrow(df))

    # Summary by prior
    println("\n" * "=" ^ 80)
    println("Summary by Prior Belief")
    println("=" ^ 80)

    gdf = groupby(df, :prior_name)
    summary_prior = combine(gdf,
        :final_adoption => mean => :mean_adoption,
        :final_gamma => mean => :mean_gamma,
        :cooperation_gap => mean => :mean_coop_gap,
        :institution_emerged => mean => :emergence_rate,
        :self_fulfilling => mean => :sf_rate
    )

    println("\nPrior        | E[p] | Adoption | Gamma | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 80)

    for pname in ["Pessimistic", "Neutral", "Optimistic"]
        row = filter(r -> r.prior_name == pname, summary_prior)[1, :]
        ep = pname == "Pessimistic" ? 0.25 : pname == "Neutral" ? 0.50 : 0.75
        @printf("%-12s | %.2f |  %5.1f%% | %.2f  |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            pname,
            ep,
            row.mean_adoption * 100,
            row.mean_gamma,
            row.emergence_rate * 100,
            row.sf_rate * 100,
            row.mean_coop_gap * 100
        )
    end

    # Summary by prior and game
    println("\n" * "=" ^ 80)
    println("Summary by Prior and Game Type")
    println("=" ^ 80)

    gdf2 = groupby(df, [:prior_name, :game_type])
    summary_full = combine(gdf2,
        :final_adoption => mean => :mean_adoption,
        :cooperation_gap => mean => :mean_coop_gap,
        :self_fulfilling => mean => :sf_rate
    )

    println("\nPrior        | Game | Adoption | Self-Fulfill | Coop Gap")
    println("-" ^ 65)

    for pname in ["Pessimistic", "Neutral", "Optimistic"]
        for gtype in ["PrisonersDilemma", "StagHunt"]
            rows = filter(r -> r.prior_name == pname && contains(r.game_type, gtype), summary_full)
            if !isempty(rows)
                row = rows[1, :]
                game_short = contains(gtype, "Prisoners") ? "PD" : "SH"
                @printf("%-12s | %-4s |  %5.1f%% |    %5.1f%%   | %+5.1f%%\n",
                    pname,
                    game_short,
                    row.mean_adoption * 100,
                    row.sf_rate * 100,
                    row.mean_coop_gap * 100
                )
            end
        end
    end

    # Theoretical analysis
    println("\n" * "=" ^ 80)
    println("Theoretical Analysis")
    println("=" ^ 80)

    println("""
    Predictions vs Results:

    1. Pessimistic priors (E[p]=0.25):
       - Agents start believing others will defect
       - Harder to build cooperation, even with bias
       - BUT: may be more sensitive to ANY positive signal

    2. Neutral priors (E[p]=0.50):
       - Baseline condition
       - Most flexible, beliefs can move either way

    3. Optimistic priors (E[p]=0.75):
       - Agents start believing others will cooperate
       - Cooperation may bootstrap faster
       - BUT: may overlook true differences in behavior
    """)

    return df
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment7()
end

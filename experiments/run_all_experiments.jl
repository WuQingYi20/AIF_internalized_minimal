"""
    Run All Experiments

Master script to execute all 8 parameter exploration experiments.
Total: ~4260 runs with 20 repeats each.

Usage:
    julia --project=. experiments/run_all_experiments.jl [experiment_numbers...]

Examples:
    julia --project=. experiments/run_all_experiments.jl          # Run all
    julia --project=. experiments/run_all_experiments.jl 1 2      # Run exp1 and exp2
    julia --project=. experiments/run_all_experiments.jl 6        # Run exp6 only
"""

using Pkg
Pkg.activate(".")

using Dates

# Experiment scripts
const EXPERIMENTS = [
    "exp1_minimal_trigger.jl",
    "exp2_complexity_penalty.jl",
    "exp3_action_precision.jl",
    "exp4_internalization.jl",
    "exp5_population_scale.jl",
    "exp6_game_types.jl",
    "exp7_prior_beliefs.jl",
    "exp8_long_term.jl"
]

const EXPERIMENT_NAMES = [
    "Minimal Trigger Conditions",
    "Complexity Penalty Sensitivity",
    "Action Precision Effect",
    "Internalization Dynamics",
    "Population Scale Effect",
    "Game Type Comparison",
    "Prior Belief Effect",
    "Long-term Stability"
]

const ESTIMATED_RUNS = [960, 420, 300, 720, 300, 720, 360, 480]

function print_menu()
    println("=" ^ 80)
    println("Parameter Exploration Experiments")
    println("=" ^ 80)
    println()

    total_runs = 0
    for (i, (name, runs)) in enumerate(zip(EXPERIMENT_NAMES, ESTIMATED_RUNS))
        @printf("  %d. %-35s (%d runs)\n", i, name, runs)
        total_runs += runs
    end

    println()
    println("Total: $total_runs runs")
    println("=" ^ 80)
end

function run_experiment(exp_num::Int)
    if exp_num < 1 || exp_num > length(EXPERIMENTS)
        @warn "Invalid experiment number: $exp_num"
        return false
    end

    exp_file = EXPERIMENTS[exp_num]
    exp_name = EXPERIMENT_NAMES[exp_num]
    est_runs = ESTIMATED_RUNS[exp_num]

    println("\n" * "=" ^ 80)
    println("Starting Experiment $exp_num: $exp_name")
    println("Estimated runs: $est_runs")
    println("Start time: $(Dates.now())")
    println("=" ^ 80)

    start_time = time()

    try
        include(joinpath(@__DIR__, exp_file))

        # Call the run function
        func_name = Symbol("run_experiment$exp_num")
        if isdefined(Main, func_name)
            getfield(Main, func_name)()
        end

        elapsed = time() - start_time
        println("\n" * "=" ^ 80)
        println("Completed Experiment $exp_num: $exp_name")
        println("Elapsed time: $(round(elapsed/60, digits=1)) minutes")
        println("=" ^ 80)

        return true
    catch e
        println("\n" * "=" ^ 80)
        println("FAILED Experiment $exp_num: $exp_name")
        println("Error: $e")
        println("=" ^ 80)
        return false
    end
end

function main()
    print_menu()

    # Parse command line arguments
    if length(ARGS) == 0
        # Run all experiments
        experiments_to_run = collect(1:8)
        println("\nRunning ALL experiments...")
    else
        # Run specified experiments
        experiments_to_run = parse.(Int, ARGS)
        println("\nRunning experiments: $experiments_to_run")
    end

    println("Press Enter to continue or Ctrl+C to cancel...")
    readline()

    # Track results
    results = Dict{Int, Bool}()
    total_start = time()

    for exp_num in experiments_to_run
        results[exp_num] = run_experiment(exp_num)
    end

    # Summary
    total_elapsed = time() - total_start

    println("\n" * "=" ^ 80)
    println("EXECUTION SUMMARY")
    println("=" ^ 80)
    println("\nTotal elapsed time: $(round(total_elapsed/60, digits=1)) minutes")
    println("\nResults:")

    for exp_num in experiments_to_run
        status = results[exp_num] ? "SUCCESS" : "FAILED"
        println("  Experiment $exp_num: $status")
    end

    # Run analysis if all succeeded
    if all(values(results))
        println("\nAll experiments completed successfully!")
        println("Run analyze_results.jl to generate comprehensive report.")
    end
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

"""
    Analyze Results

Comprehensive analysis and visualization of experiment results.
Includes:
- Heatmaps of parameter combinations
- Phase transition identification
- Sensitivity analysis
- Statistical tests
"""

using Pkg
Pkg.activate(".")

using ArbitraryInstitutions
using DataFrames
using CSV
using Statistics
using Printf

# Try to load plotting libraries (optional)
const HAS_PLOTS = try
    using CairoMakie
    true
catch
    @warn "CairoMakie not available. Plots will not be generated."
    false
end

# ============================================================
# Data Loading
# ============================================================

"""
Load all results for a given experiment.
"""
function load_experiment_results(experiment_id::String;
                                  results_dir::String = "experiments/results")
    files = filter(f -> startswith(f, experiment_id) && endswith(f, ".csv"),
                   readdir(results_dir))

    if isempty(files)
        error("No results found for: $experiment_id in $results_dir")
    end

    dfs = DataFrame[]
    for f in files
        try
            df = CSV.read(joinpath(results_dir, f), DataFrame)
            push!(dfs, df)
        catch e
            @warn "Failed to load $f: $e"
        end
    end

    return vcat(dfs...)
end

"""
Load all experiment results.
"""
function load_all_results(; results_dir::String = "experiments/results")
    experiments = ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8"]

    all_results = Dict{String, DataFrame}()

    for exp in experiments
        try
            df = load_experiment_results(exp, results_dir=results_dir)
            all_results[exp] = df
            println("Loaded $(exp): $(nrow(df)) rows")
        catch
            println("No results for $(exp)")
        end
    end

    return all_results
end

# ============================================================
# Summary Statistics
# ============================================================

"""
Compute comprehensive summary statistics.
"""
function compute_summary(df::DataFrame, group_cols::Vector{Symbol})
    gdf = groupby(df, group_cols)

    summary = combine(gdf,
        # Mean outcomes
        :final_adoption => mean => :mean_adoption,
        :final_gamma => mean => :mean_gamma,
        :cooperation_gap => mean => :mean_coop_gap,
        :belief_difference => mean => :mean_belief_diff,

        # Standard deviations
        :final_adoption => std => :std_adoption,
        :cooperation_gap => std => :std_coop_gap,

        # Success rates
        :institution_emerged => mean => :emergence_rate,
        :self_fulfilling => mean => :sf_rate,

        # Sample size
        nrow => :n
    )

    return summary
end

"""
Print formatted summary table.
"""
function print_summary(summary::DataFrame, title::String)
    println("\n" * "=" ^ 80)
    println(title)
    println("=" ^ 80)

    # Get column names for display
    cols = names(summary)

    # Print header
    for col in cols
        print(rpad(string(col), 12))
    end
    println()
    println("-" ^ (12 * length(cols)))

    # Print rows
    for row in eachrow(summary)
        for col in cols
            val = row[col]
            if val isa Float64
                print(rpad(@sprintf("%.3f", val), 12))
            else
                print(rpad(string(val), 12))
            end
        end
        println()
    end
end

# ============================================================
# Statistical Tests
# ============================================================

"""
Compute effect size (Cohen's d) between two groups.
"""
function cohens_d(group1::Vector{Float64}, group2::Vector{Float64})
    n1, n2 = length(group1), length(group2)
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)

    # Pooled standard deviation
    s_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))

    return (m1 - m2) / s_pooled
end

"""
Simple t-test (two-sample, unequal variance).
"""
function t_test(group1::Vector{Float64}, group2::Vector{Float64})
    n1, n2 = length(group1), length(group2)
    m1, m2 = mean(group1), mean(group2)
    s1, s2 = std(group1), std(group2)

    # Welch's t-test
    se = sqrt(s1^2/n1 + s2^2/n2)
    t = (m1 - m2) / se

    # Approximate degrees of freedom (Welch-Satterthwaite)
    num = (s1^2/n1 + s2^2/n2)^2
    denom = (s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1)
    df = num / denom

    return (t=t, df=df, mean_diff=m1-m2, se=se)
end

"""
Compare conditions and report statistics.
"""
function compare_conditions(df::DataFrame, condition_col::Symbol,
                           outcome_col::Symbol, conditions::Vector)
    println("\nComparison: $outcome_col by $condition_col")
    println("-" ^ 60)

    for i in 1:length(conditions)-1
        for j in i+1:length(conditions)
            c1, c2 = conditions[i], conditions[j]

            g1 = df[df[!, condition_col] .== c1, outcome_col]
            g2 = df[df[!, condition_col] .== c2, outcome_col]

            if length(g1) > 1 && length(g2) > 1
                d = cohens_d(g1, g2)
                test = t_test(g1, g2)

                effect_size = abs(d) < 0.2 ? "negligible" :
                              abs(d) < 0.5 ? "small" :
                              abs(d) < 0.8 ? "medium" : "large"

                @printf("  %s vs %s: d=%.2f (%s), t=%.2f, diff=%.3f±%.3f\n",
                    string(c1), string(c2), d, effect_size,
                    test.t, test.mean_diff, test.se)
            end
        end
    end
end

# ============================================================
# Phase Transition Analysis
# ============================================================

"""
Identify critical threshold where self-fulfilling rate exceeds 50%.
"""
function find_critical_threshold(df::DataFrame, param_col::Symbol)
    # Group by parameter and compute self-fulfilling rate
    gdf = groupby(df, param_col)
    summary = combine(gdf, :self_fulfilling => mean => :sf_rate)
    sort!(summary, param_col)

    # Find first value where sf_rate > 0.5
    threshold_idx = findfirst(summary.sf_rate .> 0.5)

    if threshold_idx === nothing
        return (threshold=nothing, above_50=false, max_rate=maximum(summary.sf_rate))
    elseif threshold_idx == 1
        return (threshold=summary[1, param_col], above_50=true,
                max_rate=maximum(summary.sf_rate))
    else
        # Interpolate between adjacent points
        p1, r1 = summary[threshold_idx-1, param_col], summary.sf_rate[threshold_idx-1]
        p2, r2 = summary[threshold_idx, param_col], summary.sf_rate[threshold_idx]

        # Linear interpolation
        threshold = p1 + (0.5 - r1) * (p2 - p1) / (r2 - r1)
        return (threshold=threshold, above_50=true, max_rate=maximum(summary.sf_rate))
    end
end

"""
Analyze phase transitions across all parameters.
"""
function analyze_phase_transitions(df::DataFrame)
    println("\n" * "=" ^ 80)
    println("Phase Transition Analysis")
    println("=" ^ 80)

    params = [:bias, :complexity_penalty, :action_precision,
              :n_agents, :γ_update_factor]

    for param in params
        if param in names(df)
            unique_vals = unique(df[!, param])
            if length(unique_vals) >= 3
                result = find_critical_threshold(df, param)
                println("\n$param:")
                if result.above_50
                    println("  Critical threshold: $(round(result.threshold, digits=4))")
                else
                    println("  No threshold found (max rate: $(round(result.max_rate*100, digits=1))%)")
                end
            end
        end
    end
end

# ============================================================
# Sensitivity Analysis
# ============================================================

"""
Compute sensitivity of outcome to each parameter.
Uses variance decomposition.
"""
function sensitivity_analysis(df::DataFrame, outcome_col::Symbol)
    println("\n" * "=" ^ 80)
    println("Sensitivity Analysis: $outcome_col")
    println("=" ^ 80)

    # Total variance
    total_var = var(df[!, outcome_col])
    println("\nTotal variance: $(round(total_var, digits=4))")

    # Variance explained by each parameter
    params = [:bias, :complexity_penalty, :action_precision,
              :n_agents, :γ_update_factor, :initial_precision, :max_precision]

    sensitivities = DataFrame(parameter=String[], variance_explained=Float64[],
                             pct_variance=Float64[])

    for param in params
        if param in names(df)
            unique_vals = unique(df[!, param])
            if length(unique_vals) >= 2
                # Between-group variance
                gdf = groupby(df, param)
                group_means = [mean(g[!, outcome_col]) for g in gdf]
                group_sizes = [nrow(g) for g in gdf]
                grand_mean = mean(df[!, outcome_col])

                between_var = sum(n * (m - grand_mean)^2
                                  for (n, m) in zip(group_sizes, group_means)) /
                              sum(group_sizes)

                pct = between_var / total_var * 100

                push!(sensitivities, (string(param), between_var, pct))
            end
        end
    end

    sort!(sensitivities, :variance_explained, rev=true)

    println("\nParameter sensitivity ranking:")
    println("-" ^ 50)
    @printf("%-20s | %s | %s\n", "Parameter", "Var Explained", "% of Total")
    println("-" ^ 50)

    for row in eachrow(sensitivities)
        @printf("%-20s | %.4f       | %5.1f%%\n",
            row.parameter, row.variance_explained, row.pct_variance)
    end

    return sensitivities
end

# ============================================================
# Plotting (if available)
# ============================================================

if HAS_PLOTS

"""
Create heatmap of self-fulfilling rate.
"""
function plot_heatmap(df::DataFrame, x_col::Symbol, y_col::Symbol;
                      outcome_col::Symbol = :self_fulfilling,
                      title::String = "Self-Fulfilling Rate")
    # Aggregate
    gdf = groupby(df, [x_col, y_col])
    summary = combine(gdf, outcome_col => mean => :value)

    # Create matrix
    x_vals = sort(unique(summary[!, x_col]))
    y_vals = sort(unique(summary[!, y_col]))

    matrix = zeros(length(y_vals), length(x_vals))
    for row in eachrow(summary)
        xi = findfirst(x_vals .== row[x_col])
        yi = findfirst(y_vals .== row[y_col])
        matrix[yi, xi] = row.value
    end

    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
        xlabel=string(x_col),
        ylabel=string(y_col),
        title=title
    )

    hm = heatmap!(ax, 1:length(x_vals), 1:length(y_vals), matrix,
                  colormap=:viridis)
    Colorbar(fig[1, 2], hm, label="Rate")

    ax.xticks = (1:length(x_vals), string.(x_vals))
    ax.yticks = (1:length(y_vals), string.(y_vals))

    return fig
end

"""
Plot emergence rate trajectory over parameter values.
"""
function plot_trajectory(df::DataFrame, param_col::Symbol;
                         outcome_col::Symbol = :self_fulfilling,
                         title::String = "")
    gdf = groupby(df, param_col)
    summary = combine(gdf,
        outcome_col => mean => :mean_val,
        outcome_col => std => :std_val,
        nrow => :n
    )
    sort!(summary, param_col)

    fig = Figure(size=(800, 500))
    ax = Axis(fig[1, 1],
        xlabel=string(param_col),
        ylabel=string(outcome_col),
        title=isempty(title) ? "$(outcome_col) vs $(param_col)" : title
    )

    # Confidence bands
    se = summary.std_val ./ sqrt.(summary.n)
    band!(ax, summary[!, param_col],
          summary.mean_val .- 1.96 .* se,
          summary.mean_val .+ 1.96 .* se,
          alpha=0.3)

    # Mean line
    lines!(ax, summary[!, param_col], summary.mean_val, linewidth=2)
    scatter!(ax, summary[!, param_col], summary.mean_val, markersize=10)

    # 50% threshold line
    hlines!(ax, [0.5], linestyle=:dash, color=:red, alpha=0.5)

    return fig
end

"""
Save all plots for an experiment.
"""
function save_experiment_plots(df::DataFrame, experiment_id::String;
                               output_dir::String = "experiments/figures")
    mkpath(output_dir)

    # Determine which plots to create based on available columns
    if :bias in names(df) && :bias_duration in names(df)
        fig = plot_heatmap(df, :bias, :bias_duration,
                          title="Self-Fulfilling Rate: Bias × Duration")
        save(joinpath(output_dir, "$(experiment_id)_heatmap.png"), fig)
    end

    if :bias in names(df)
        fig = plot_trajectory(df, :bias, title="Emergence vs Bias")
        save(joinpath(output_dir, "$(experiment_id)_bias_trajectory.png"), fig)
    end

    println("Plots saved to $output_dir")
end

end  # if HAS_PLOTS

# ============================================================
# Main Analysis Functions
# ============================================================

"""
Run full analysis on loaded results.
"""
function analyze_experiment(df::DataFrame, experiment_id::String)
    println("\n" * "=" ^ 80)
    println("ANALYSIS: $experiment_id")
    println("=" ^ 80)
    println("Total runs: $(nrow(df))")
    println("Unique configurations: $(length(unique(df.config_id)))")

    # Basic statistics
    println("\nOverall Statistics:")
    println("  Mean adoption: $(round(mean(df.final_adoption)*100, digits=1))%")
    println("  Mean gamma: $(round(mean(df.final_gamma), digits=2))")
    println("  Emergence rate: $(round(mean(df.institution_emerged)*100, digits=1))%")
    println("  Self-fulfilling rate: $(round(mean(df.self_fulfilling)*100, digits=1))%")
    println("  Mean coop gap: $(round(mean(df.cooperation_gap)*100, digits=1))%")

    # Phase transition analysis
    analyze_phase_transitions(df)

    # Sensitivity analysis
    sensitivity_analysis(df, :self_fulfilling)

    # Generate plots if possible
    if HAS_PLOTS
        try
            save_experiment_plots(df, experiment_id)
        catch e
            @warn "Failed to generate plots: $e"
        end
    end
end

"""
Generate comprehensive report across all experiments.
"""
function generate_full_report(; results_dir::String = "experiments/results")
    println("=" ^ 80)
    println("COMPREHENSIVE ANALYSIS REPORT")
    println("Generated: $(Dates.now())")
    println("=" ^ 80)

    all_results = load_all_results(results_dir=results_dir)

    for (exp_id, df) in sort(collect(all_results))
        analyze_experiment(df, exp_id)
    end

    # Cross-experiment summary
    println("\n" * "=" ^ 80)
    println("CROSS-EXPERIMENT SUMMARY")
    println("=" ^ 80)

    summary_data = DataFrame(
        experiment = String[],
        n_runs = Int[],
        emergence_rate = Float64[],
        sf_rate = Float64[],
        mean_coop_gap = Float64[]
    )

    for (exp_id, df) in sort(collect(all_results))
        push!(summary_data, (
            exp_id,
            nrow(df),
            mean(df.institution_emerged),
            mean(df.self_fulfilling),
            mean(df.cooperation_gap)
        ))
    end

    println("\nExperiment     | Runs | Emergence | Self-Fulfill | Coop Gap")
    println("-" ^ 60)
    for row in eachrow(summary_data)
        @printf("%-14s | %4d |   %5.1f%%  |    %5.1f%%   | %+5.1f%%\n",
            row.experiment,
            row.n_runs,
            row.emergence_rate * 100,
            row.sf_rate * 100,
            row.mean_coop_gap * 100
        )
    end
end

# ============================================================
# Run Analysis
# ============================================================

println("Analysis framework loaded.")
println()
println("Usage:")
println("  df = load_experiment_results(\"exp1_minimal_trigger\")")
println("  analyze_experiment(df, \"exp1\")")
println("  generate_full_report()")

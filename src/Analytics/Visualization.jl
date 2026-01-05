"""
    Visualization

Real-time visualization dashboard using GLMakie.
Provides live updating plots during simulation.
"""
module Visualization

using Statistics
using ..BrainTypes
using ..WorldTypes
using ..Convergence

export create_live_dashboard, update_dashboard!, plot_internalization_matrix,
       plot_adoption_trajectory, plot_belief_evolution,
       DashboardState

# Note: GLMakie is loaded conditionally to avoid dependency issues
# Users should call `using GLMakie` before using visualization functions

"""
    DashboardState

Mutable state for the live dashboard.
Stores time series data for plotting.
"""
mutable struct DashboardState
    steps::Vector{Int}
    adoption_rates::Vector{Float64}
    mean_internalizations::Vector{Float64}
    label_correlations::Vector{Float64}
    cooperation_gaps::Vector{Float64}
    internalization_matrix::Matrix{Float64}  # agents × time

    function DashboardState(n_agents::Int, max_steps::Int=1000)
        new(
            Int[],
            Float64[],
            Float64[],
            Float64[],
            Float64[],
            zeros(n_agents, 0)
        )
    end
end

"""
    record_step!(state, model, step)

Record metrics for the current step.
"""
function record_step!(state::DashboardState, model, step::Int)
    push!(state.steps, step)
    push!(state.adoption_rates, institutional_adoption_rate(model))
    push!(state.mean_internalizations, mean_internalization(model))
    push!(state.label_correlations, label_correlation(model))
    push!(state.cooperation_gaps, cooperation_gap(model))

    # Update internalization matrix
    γ_values = [a.cognitive_state.γ for a in sort(collect(allagents(model)), by=a->a.id)]
    state.internalization_matrix = hcat(state.internalization_matrix, γ_values)
end

"""
    create_live_dashboard(sim; backend=:gl)

Create a live updating dashboard for the simulation.
Requires GLMakie to be loaded.

Returns (figure, observables, update_function)
"""
function create_live_dashboard(sim; backend::Symbol=:gl)
    # Check if Makie is available
    if !isdefined(Main, :GLMakie) && !isdefined(Main, :CairoMakie)
        @warn "No Makie backend loaded. Please run `using GLMakie` or `using CairoMakie` first."
        return nothing
    end

    Makie = if isdefined(Main, :GLMakie)
        Main.GLMakie
    else
        Main.CairoMakie
    end

    n_agents = length(collect(allagents(sim.model)))

    # Create figure with 2×2 layout
    fig = Makie.Figure(size=(1200, 800))

    # Observable data sources
    adoption_obs = Makie.Observable(Float64[0.0])
    internalization_obs = Makie.Observable(Float64[0.0])
    correlation_obs = Makie.Observable(Float64[0.0])
    steps_obs = Makie.Observable(Int[0])
    heatmap_obs = Makie.Observable(zeros(n_agents, 1))

    # Panel 1: Institutional adoption rate over time
    ax1 = Makie.Axis(fig[1, 1],
        title = "Institutional Adoption Rate",
        xlabel = "Step",
        ylabel = "Adoption Rate",
        limits = (nothing, nothing, 0, 1)
    )
    Makie.lines!(ax1, steps_obs, adoption_obs, color=:blue, linewidth=2)
    Makie.hlines!(ax1, [0.5], color=:gray, linestyle=:dash)

    # Panel 2: Internalization heatmap (agents × time)
    ax2 = Makie.Axis(fig[1, 2],
        title = "Internalization Depth (γ)",
        xlabel = "Step",
        ylabel = "Agent ID"
    )
    Makie.heatmap!(ax2, heatmap_obs, colormap=:viridis)

    # Panel 3: Label-behavior correlation
    ax3 = Makie.Axis(fig[2, 1],
        title = "Label-Behavior Correlation",
        xlabel = "Step",
        ylabel = "Correlation",
        limits = (nothing, nothing, -1, 1)
    )
    Makie.lines!(ax3, steps_obs, correlation_obs, color=:red, linewidth=2)
    Makie.hlines!(ax3, [0.0], color=:gray, linestyle=:dash)

    # Panel 4: Mean internalization over time
    ax4 = Makie.Axis(fig[2, 2],
        title = "Mean Internalization",
        xlabel = "Step",
        ylabel = "Mean γ"
    )
    Makie.lines!(ax4, steps_obs, internalization_obs, color=:green, linewidth=2)

    # Create dashboard state
    state = DashboardState(n_agents)

    # Update function
    function update!(step::Int)
        record_step!(state, sim.model, step)

        steps_obs[] = state.steps
        adoption_obs[] = state.adoption_rates
        internalization_obs[] = state.mean_internalizations
        correlation_obs[] = state.label_correlations
        heatmap_obs[] = state.internalization_matrix

        # Trigger plot updates
        Makie.notify(steps_obs)
        Makie.notify(adoption_obs)
        Makie.notify(internalization_obs)
        Makie.notify(correlation_obs)
        Makie.notify(heatmap_obs)
    end

    return (fig, state, update!)
end

"""
    plot_internalization_matrix(sim_or_state; backend=:cairo)

Create a static heatmap of internalization over time.
Useful for post-hoc analysis.
"""
function plot_internalization_matrix(state::DashboardState; kwargs...)
    if !isdefined(Main, :CairoMakie) && !isdefined(Main, :GLMakie)
        @warn "No Makie backend loaded."
        return nothing
    end

    Makie = isdefined(Main, :CairoMakie) ? Main.CairoMakie : Main.GLMakie

    fig = Makie.Figure(size=(800, 600))
    ax = Makie.Axis(fig[1, 1],
        title = "Internalization Depth Over Time",
        xlabel = "Simulation Step",
        ylabel = "Agent ID"
    )

    hm = Makie.heatmap!(ax, state.internalization_matrix,
        colormap = :viridis,
        colorrange = (0, maximum(state.internalization_matrix) * 1.1)
    )

    Makie.Colorbar(fig[1, 2], hm, label = "γ (Internalization)")

    return fig
end

"""
    plot_adoption_trajectory(state)

Plot the institutional adoption rate over time.
"""
function plot_adoption_trajectory(state::DashboardState)
    if !isdefined(Main, :CairoMakie) && !isdefined(Main, :GLMakie)
        @warn "No Makie backend loaded."
        return nothing
    end

    Makie = isdefined(Main, :CairoMakie) ? Main.CairoMakie : Main.GLMakie

    fig = Makie.Figure(size=(800, 400))
    ax = Makie.Axis(fig[1, 1],
        title = "Institutional Model Adoption Over Time",
        xlabel = "Simulation Step",
        ylabel = "Fraction of Agents",
        limits = (nothing, nothing, 0, 1)
    )

    Makie.lines!(ax, state.steps, state.adoption_rates,
        color = :blue, linewidth = 2, label = "Institutional")

    neutral_rates = 1.0 .- state.adoption_rates
    Makie.lines!(ax, state.steps, neutral_rates,
        color = :gray, linewidth = 2, label = "Neutral")

    Makie.axislegend(ax)

    return fig
end

"""
    plot_belief_evolution(sim)

Plot how agent beliefs about ingroup/outgroup cooperation evolve.
"""
function plot_belief_evolution(sim)
    if !isdefined(Main, :CairoMakie) && !isdefined(Main, :GLMakie)
        @warn "No Makie backend loaded."
        return nothing
    end

    Makie = isdefined(Main, :CairoMakie) ? Main.CairoMakie : Main.GLMakie

    fig = Makie.Figure(size=(1000, 400))

    # Collect current beliefs
    agents = sort(collect(allagents(sim.model)), by=a->a.id)

    ingroup_beliefs = Float64[]
    outgroup_beliefs = Float64[]
    agent_ids = Int[]

    for agent in agents
        beliefs = agent.cognitive_state.beliefs
        push!(agent_ids, agent.id)
        push!(ingroup_beliefs, beliefs.α_ingroup / (beliefs.α_ingroup + beliefs.β_ingroup))
        push!(outgroup_beliefs, beliefs.α_outgroup / (beliefs.α_outgroup + beliefs.β_outgroup))
    end

    ax = Makie.Axis(fig[1, 1],
        title = "Agent Beliefs About Cooperation",
        xlabel = "Agent ID",
        ylabel = "Believed Cooperation Rate",
        xticks = agent_ids
    )

    Makie.scatter!(ax, agent_ids, ingroup_beliefs,
        color = :blue, markersize = 15, label = "Ingroup")
    Makie.scatter!(ax, agent_ids, outgroup_beliefs,
        color = :red, markersize = 15, label = "Outgroup")

    Makie.axislegend(ax, position = :rt)

    return fig
end

"""
    create_summary_figure(state)

Create a comprehensive summary figure with all key metrics.
"""
function create_summary_figure(state::DashboardState)
    if !isdefined(Main, :CairoMakie) && !isdefined(Main, :GLMakie)
        @warn "No Makie backend loaded."
        return nothing
    end

    Makie = isdefined(Main, :CairoMakie) ? Main.CairoMakie : Main.GLMakie

    fig = Makie.Figure(size=(1400, 1000))

    # Title
    Makie.Label(fig[0, 1:2], "Institution Emergence Simulation Summary",
        fontsize = 24, tellwidth = false)

    # Panel 1: Adoption trajectory
    ax1 = Makie.Axis(fig[1, 1],
        title = "Model Adoption",
        xlabel = "Step",
        ylabel = "Rate"
    )
    Makie.lines!(ax1, state.steps, state.adoption_rates, color=:blue, linewidth=2)

    # Panel 2: Internalization heatmap
    ax2 = Makie.Axis(fig[1, 2],
        title = "Internalization Depth (γ)",
        xlabel = "Step",
        ylabel = "Agent"
    )
    hm = Makie.heatmap!(ax2, state.internalization_matrix, colormap=:viridis)
    Makie.Colorbar(fig[1, 3], hm)

    # Panel 3: Correlation evolution
    ax3 = Makie.Axis(fig[2, 1],
        title = "Label-Behavior Correlation",
        xlabel = "Step",
        ylabel = "Correlation"
    )
    Makie.lines!(ax3, state.steps, state.label_correlations, color=:red, linewidth=2)
    Makie.hlines!(ax3, [0], color=:gray, linestyle=:dash)

    # Panel 4: Cooperation gap
    ax4 = Makie.Axis(fig[2, 2],
        title = "Cooperation Gap (Ingroup - Outgroup)",
        xlabel = "Step",
        ylabel = "Gap"
    )
    Makie.lines!(ax4, state.steps, state.cooperation_gaps, color=:orange, linewidth=2)
    Makie.hlines!(ax4, [0], color=:gray, linestyle=:dash)

    return fig
end

"""
    save_dashboard(state, filename; format=:png)

Save the current dashboard state as an image.
"""
function save_dashboard(state::DashboardState, filename::String; format::Symbol=:png)
    fig = create_summary_figure(state)
    if fig !== nothing
        Makie = isdefined(Main, :CairoMakie) ? Main.CairoMakie : Main.GLMakie
        Makie.save(filename, fig)
        @info "Dashboard saved to $filename"
    end
end

end # module

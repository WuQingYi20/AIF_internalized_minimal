using ArbitraryInstitutions

println("=" ^ 60)
println("Arbitrary Institutions - Multi-Scenario Experiments")
println("=" ^ 60)

# Experiment 1: Different Game Types
println("\n### Experiment 1: Different Game Types ###\n")

for (name, game) in [
    ("Prisoner's Dilemma", PrisonersDilemma()),
    ("Stag Hunt", StagHunt()),
    ("Harmony", Harmony())
]
    sim = Simulation(
        n_agents = 16,
        game_type = game,
        complexity_penalty = 0.05,
        seed = 42
    )
    run_evolution!(sim, 300)

    println("$name:")
    println("  Adoption: $(round(institutional_adoption_rate(sim)*100, digits=1))%")
    println("  Mean γ: $(round(mean_internalization(sim), digits=2))")
    println("  Correlation: $(round(label_correlation(sim), digits=3))")
    println()
end

# Experiment 2: Complexity Penalty Sweep
println("\n### Experiment 2: Complexity Penalty Sweep ###\n")

for penalty in [0.001, 0.01, 0.05, 0.1, 0.2]
    sim = Simulation(
        n_agents = 16,
        complexity_penalty = penalty,
        seed = 42
    )
    run_evolution!(sim, 300)

    println("Penalty=$penalty: Adoption=$(round(institutional_adoption_rate(sim)*100, digits=1))%, γ=$(round(mean_internalization(sim), digits=2))")
end

# Experiment 3: Population Size Effect
println("\n\n### Experiment 3: Population Size Effect ###\n")

for n in [8, 16, 32, 64]
    sim = Simulation(
        n_agents = n,
        complexity_penalty = 0.05,
        seed = 42
    )
    run_evolution!(sim, 300)

    println("N=$n agents: Adoption=$(round(institutional_adoption_rate(sim)*100, digits=1))%, γ=$(round(mean_internalization(sim), digits=2))")
end

# Experiment 4: Long-run dynamics
println("\n\n### Experiment 4: Long-run Dynamics (1000 steps) ###\n")

sim = Simulation(n_agents=16, complexity_penalty=0.02, seed=999)

for step in [100, 200, 300, 500, 700, 1000]
    target = step - sim.step_count
    if target > 0
        run_evolution!(sim, target)
    end
    println("Step $step: Adoption=$(round(institutional_adoption_rate(sim)*100, digits=1))%, γ=$(round(mean_internalization(sim), digits=2)), Corr=$(round(label_correlation(sim), digits=3))")
end

println("\n" * "=" ^ 60)
println("Experiments Complete!")
println("=" ^ 60)

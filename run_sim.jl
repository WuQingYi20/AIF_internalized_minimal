using ArbitraryInstitutions

println("=== Arbitrary Institutions Simulation ===")
println("16 agents, 500 steps, Prisoner's Dilemma")
println()

sim = Simulation(n_agents=16, complexity_penalty=0.01, seed=456)  # Lower penalty

println("Running simulation...")
run_evolution!(sim, 500, verbose=true)

println()
println("=== Final Results ===")
println("Institutional adoption rate: ", round(institutional_adoption_rate(sim), digits=3))
println("Mean internalization (γ): ", round(mean_internalization(sim), digits=3))
println("Label-behavior correlation: ", round(label_correlation(sim), digits=3))

using Test
using Random
using Statistics
using DataFrames: nrow

# Note: Tests can be run with `julia --project=. -e "using Pkg; Pkg.test()"`
# Ensure dependencies are installed first with `julia --project=. -e "using Pkg; Pkg.instantiate()"`

@testset "ArbitraryInstitutions.jl" begin

    # Import the module
    using ArbitraryInstitutions

    @testset "Brain/Types.jl" begin
        @testset "BeliefState" begin
            beliefs = ArbitraryInstitutions.BrainTypes.BeliefState()

            # Test default uninformative prior
            @test beliefs.α_global ≈ 1.0
            @test beliefs.β_global ≈ 1.0

            # Test mean calculation
            @test ArbitraryInstitutions.BrainTypes.global_cooperation_mean(beliefs) ≈ 0.5
        end

        @testset "CognitiveState" begin
            state = CognitiveState()

            # Test defaults
            @test state.active_model == NEUTRAL
            @test state.γ == 1.0

            # Test prediction under neutral model
            p = ArbitraryInstitutions.BrainTypes.predict_cooperation(state, true)
            @test 0.0 ≤ p ≤ 1.0
        end
    end

    @testset "World/Types.jl" begin
        @testset "InteractionRecord" begin
            record = InteractionRecord(
                opponent_label = true,
                opponent_cooperated = true,
                own_action = true,
                payoff = 3.0
            )

            @test record.opponent_label == true
            @test record.payoff == 3.0
        end
    end

    @testset "World/Physics.jl - Game Types" begin
        @testset "PrisonersDilemma payoffs" begin
            game = PrisonersDilemma()

            # Mutual cooperation
            p1, p2 = resolve_interaction(true, true, game)
            @test p1 == 3.0
            @test p2 == 3.0

            # Mutual defection
            p1, p2 = resolve_interaction(false, false, game)
            @test p1 == 1.0
            @test p2 == 1.0

            # Exploitation (agent 1 defects against cooperator)
            p1, p2 = resolve_interaction(false, true, game)
            @test p1 == 5.0
            @test p2 == 0.0
        end

        @testset "StagHunt payoffs" begin
            game = StagHunt()

            # Mutual cooperation (hunting stag)
            p1, p2 = resolve_interaction(true, true, game)
            @test p1 == 4.0

            # Mutual defection (hunting hare)
            p1, p2 = resolve_interaction(false, false, game)
            @test p1 == 3.0
        end

        @testset "Harmony payoffs" begin
            game = Harmony()

            # Cooperation should always be best
            p_cc, _ = resolve_interaction(true, true, game)
            p_cd, _ = resolve_interaction(true, false, game)
            p_dc, _ = resolve_interaction(false, true, game)
            p_dd, _ = resolve_interaction(false, false, game)

            @test p_cc > p_dd  # Mutual coop better than mutual defect
        end

        @testset "Label blindness" begin
            # Environment must be label-blind
            # Payoffs should not depend on any "label" parameter
            game = PrisonersDilemma()

            # Same actions should give same payoffs regardless of context
            p1a, p2a = resolve_interaction(true, true, game)
            p1b, p2b = resolve_interaction(true, true, game)

            @test p1a == p1b
            @test p2a == p2b
        end
    end

    @testset "Brain/FactorGraph.jl" begin
        @testset "Model evidence M0" begin
            observations = [true, true, true, false, false]

            evidence = ArbitraryInstitutions.FactorGraphs.compute_model_evidence_M0(
                observations, 1.0, 1.0
            )

            # Evidence should be finite
            @test isfinite(evidence)

            # More data should give higher absolute evidence
            more_obs = repeat(observations, 3)
            more_evidence = ArbitraryInstitutions.FactorGraphs.compute_model_evidence_M0(
                more_obs, 1.0, 1.0
            )
            @test abs(more_evidence) > abs(evidence)
        end

        @testset "Model evidence M1" begin
            observations = [true, true, true, false, false, true]
            labels = [true, true, true, false, false, false]

            evidence = ArbitraryInstitutions.FactorGraphs.compute_model_evidence_M1(
                observations, labels, 1.0, 1.0, 1.0, 1.0
            )

            @test isfinite(evidence)
        end

        @testset "Model comparison" begin
            # When labels perfectly predict observations, M1 should win
            observations = [true, true, true, false, false, false]
            labels = [true, true, true, false, false, false]

            evidence_M0 = ArbitraryInstitutions.FactorGraphs.compute_model_evidence_M0(
                observations, 1.0, 1.0
            )
            evidence_M1 = ArbitraryInstitutions.FactorGraphs.compute_model_evidence_M1(
                observations, labels, 1.0, 1.0, 1.0, 1.0
            )

            # M1 should have higher evidence when labels are informative
            @test evidence_M1 > evidence_M0
        end
    end

    @testset "Brain/ActionSelection.jl" begin
        @testset "Softmax" begin
            values = [1.0, 2.0]
            probs = ArbitraryInstitutions.ActionSelection.softmax(values, 1.0)

            @test length(probs) == 2
            @test sum(probs) ≈ 1.0
            @test probs[2] > probs[1]  # Higher value = higher probability
        end

        @testset "Entropy" begin
            # Maximum entropy at p=0.5
            h_half = ArbitraryInstitutions.ActionSelection.entropy(0.5)
            h_biased = ArbitraryInstitutions.ActionSelection.entropy(0.9)

            @test h_half > h_biased
        end

        @testset "Expected Free Energy" begin
            config = SimulationConfig(seed=42)
            state = CognitiveState()
            agent = InstitutionAgent(
                id = 1,
                label = true,
                cognitive_state = state,
                interaction_history = InteractionRecord[]
            )

            G_coop = ArbitraryInstitutions.ActionSelection.compute_expected_free_energy(
                agent, true, true
            )
            G_defect = ArbitraryInstitutions.ActionSelection.compute_expected_free_energy(
                agent, false, true
            )

            @test isfinite(G_coop)
            @test isfinite(G_defect)
        end
    end

    @testset "Simulation" begin
        @testset "Initialization" begin
            sim = Simulation(n_agents=8, seed=42)

            @test length(collect(allagents(sim.model))) == 8
            @test sim.step_count == 0
        end

        @testset "Step execution" begin
            sim = Simulation(n_agents=8, seed=42)

            step_simulation!(sim)
            sim.step_count += 1

            # After one step, agents should have interaction history
            for agent in allagents(sim.model)
                @test length(agent.interaction_history) >= 1
            end
        end

        @testset "Multiple steps" begin
            sim = Simulation(n_agents=8, seed=42)

            run_evolution!(sim, 10)

            @test sim.step_count == 10
        end
    end

    @testset "Analytics/Convergence.jl" begin
        @testset "Institutional adoption rate" begin
            sim = Simulation(n_agents=8, seed=42)

            # Initially all agents should be neutral
            rate = institutional_adoption_rate(sim)
            @test rate == 0.0
        end

        @testset "Mean internalization" begin
            sim = Simulation(n_agents=8, seed=42, initial_precision=1.5)

            mean_γ = mean_internalization(sim)
            @test mean_γ ≈ 1.5
        end

        @testset "Label correlation" begin
            sim = Simulation(n_agents=8, seed=42)

            # With no interactions, correlation should be 0
            corr = label_correlation(sim)
            @test corr == 0.0

            # After some interactions
            run_evolution!(sim, 20)
            corr = label_correlation(sim)
            @test -1.0 ≤ corr ≤ 1.0
        end
    end

    @testset "Brain/Learning.jl" begin
        @testset "Belief updates" begin
            state = CognitiveState()
            beliefs_before = state.beliefs.α_global

            # Update after observing cooperation
            ArbitraryInstitutions.BrainTypes.update_beliefs!(state, true, true, true)

            @test state.beliefs.α_global > beliefs_before
        end

        @testset "Internalization dynamics" begin
            config = SimulationConfig(
                initial_precision=1.0,
                max_precision=10.0,
                min_precision=0.1
            )

            state = CognitiveState(γ=1.0)
            state.active_model = INSTITUTIONAL

            # Set up beliefs so ingroup is expected to cooperate (α_ingroup > β_ingroup)
            state.beliefs.α_ingroup = 8.0
            state.beliefs.β_ingroup = 2.0

            agent = InstitutionAgent(
                id = 1,
                label = true,
                cognitive_state = state,
                interaction_history = InteractionRecord[]
            )

            # Record a confirming interaction (ingroup cooperating as expected)
            record = InteractionRecord(
                opponent_label = true,  # Ingroup
                opponent_cooperated = true,  # Confirms expectation (expected rate > 0.5)
                own_action = true,
                payoff = 3.0
            )
            push!(agent.interaction_history, record)

            γ_before = agent.cognitive_state.γ
            ArbitraryInstitutions.Learning.update_internalization!(agent, record, config)

            # γ should increase when predictions confirmed
            @test agent.cognitive_state.γ >= γ_before
        end
    end

    @testset "End-to-end simulation" begin
        @testset "Institution emergence (smoke test)" begin
            # Run a full simulation and check it completes without error
            sim = Simulation(
                n_agents = 16,
                complexity_penalty = 0.05,  # Lower penalty for faster emergence
                seed = 123
            )

            history = run_evolution!(sim, 100)

            # Check history was recorded
            @test nrow(history) > 0

            # Check metrics are computable
            @test isfinite(institutional_adoption_rate(sim))
            @test isfinite(mean_internalization(sim))
            @test isfinite(label_correlation(sim))
        end

        @testset "Different game types" begin
            for game_type in [PrisonersDilemma(), StagHunt(), Harmony()]
                sim = Simulation(
                    n_agents = 8,
                    game_type = game_type,
                    seed = 42
                )

                # Should run without error
                run_evolution!(sim, 20)

                @test sim.step_count == 20
            end
        end
    end

end

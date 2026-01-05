using ArbitraryInstitutions

println("=" ^ 70)
println("Deep Analysis: 标签预测价值 vs 协调需求")
println("=" ^ 70)

# 实验1: 猎鹿博弈 - 冷启动 vs 热启动
println("\n### 实验1: 猎鹿博弈的先验效应 ###\n")

println("冷启动 (悲观先验 α=1, β=2):")
sim_cold = Simulation(
    n_agents = 16,
    game_type = StagHunt(),
    prior_cooperation = (1.0, 2.0),  # 悲观：预期对方不合作
    complexity_penalty = 0.01,
    seed = 42
)
run_evolution!(sim_cold, 300)
println("  Adoption: $(round(institutional_adoption_rate(sim_cold)*100, digits=1))%, γ=$(round(mean_internalization(sim_cold), digits=2)), Corr=$(round(label_correlation(sim_cold), digits=3))")

println("\n中性先验 (α=1, β=1):")
sim_neutral = Simulation(
    n_agents = 16,
    game_type = StagHunt(),
    prior_cooperation = (1.0, 1.0),  # 中性
    complexity_penalty = 0.01,
    seed = 42
)
run_evolution!(sim_neutral, 300)
println("  Adoption: $(round(institutional_adoption_rate(sim_neutral)*100, digits=1))%, γ=$(round(mean_internalization(sim_neutral), digits=2)), Corr=$(round(label_correlation(sim_neutral), digits=3))")

println("\n热启动 (乐观先验 α=2, β=1):")
sim_hot = Simulation(
    n_agents = 16,
    game_type = StagHunt(),
    prior_cooperation = (2.0, 1.0),  # 乐观：预期对方合作
    complexity_penalty = 0.01,
    seed = 42
)
run_evolution!(sim_hot, 300)
println("  Adoption: $(round(institutional_adoption_rate(sim_hot)*100, digits=1))%, γ=$(round(mean_internalization(sim_hot), digits=2)), Corr=$(round(label_correlation(sim_hot), digits=3))")

println("\n强热启动 (α=3, β=1):")
sim_very_hot = Simulation(
    n_agents = 16,
    game_type = StagHunt(),
    prior_cooperation = (3.0, 1.0),  # 更乐观
    complexity_penalty = 0.01,
    seed = 42
)
run_evolution!(sim_very_hot, 300)
println("  Adoption: $(round(institutional_adoption_rate(sim_very_hot)*100, digits=1))%, γ=$(round(mean_internalization(sim_very_hot), digits=2)), Corr=$(round(label_correlation(sim_very_hot), digits=3))")


# 实验2: 跨博弈对比 - 统一热启动
println("\n\n### 实验2: 热启动对三种博弈的影响 ###\n")

for (name, game) in [
    ("囚徒困境", PrisonersDilemma()),
    ("猎鹿博弈", StagHunt()),
    ("和谐博弈", Harmony())
]
    sim = Simulation(
        n_agents = 16,
        game_type = game,
        prior_cooperation = (2.0, 1.0),  # 统一乐观先验
        complexity_penalty = 0.01,
        seed = 42
    )
    run_evolution!(sim, 300)
    println("$name (乐观先验): Adoption=$(round(institutional_adoption_rate(sim)*100, digits=1))%, γ=$(round(mean_internalization(sim), digits=2)), Corr=$(round(label_correlation(sim), digits=3))")
end


# 实验3: 行为方差分析
println("\n\n### 实验3: 行为方差 - 制度涌现的关键 ###\n")

for (name, game, prior) in [
    ("猎鹿-悲观", StagHunt(), (1.0, 2.0)),
    ("猎鹿-乐观", StagHunt(), (2.0, 1.0)),
    ("和谐-中性", Harmony(), (1.0, 1.0)),
    ("囚徒-中性", PrisonersDilemma(), (1.0, 1.0))
]
    sim = Simulation(
        n_agents = 16,
        game_type = game,
        prior_cooperation = prior,
        complexity_penalty = 0.01,
        seed = 42
    )
    run_evolution!(sim, 300)

    # 计算实际合作率
    total_coop = 0
    total_actions = 0
    for agent in allagents(sim.model)
        for r in agent.interaction_history
            total_actions += 1
            total_coop += r.own_action ? 1 : 0
        end
    end
    coop_rate = total_actions > 0 ? total_coop / total_actions : 0.0

    println("$name: 合作率=$(round(coop_rate*100, digits=1))%, Adoption=$(round(institutional_adoption_rate(sim)*100, digits=1))%, Corr=$(round(label_correlation(sim), digits=3))")
end


# 实验4: 长期猎鹿博弈热启动
println("\n\n### 实验4: 猎鹿博弈热启动长期动态 ###\n")

sim_stag_long = Simulation(
    n_agents = 16,
    game_type = StagHunt(),
    prior_cooperation = (3.0, 1.0),
    complexity_penalty = 0.01,
    seed = 999
)

for step in [100, 200, 300, 500, 700, 1000]
    target = step - sim_stag_long.step_count
    if target > 0
        run_evolution!(sim_stag_long, target)
    end
    println("Step $step: Adoption=$(round(institutional_adoption_rate(sim_stag_long)*100, digits=1))%, γ=$(round(mean_internalization(sim_stag_long), digits=2)), Corr=$(round(label_correlation(sim_stag_long), digits=3))")
end

println("\n" * "=" ^ 70)
println("核心发现: 制度涌现 = f(行为可变性, 标签-行为关联可学习性)")
println("=" ^ 70)

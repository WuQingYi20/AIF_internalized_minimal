using ArbitraryInstitutions
using Printf
using Statistics: mean, std
using Random

println("=" ^ 80)
println("最小触发条件实验：多大的初始偏差能触发持续的制度涌现？")
println("=" ^ 80)

"""
带偏差的行动选择函数
- 给 label=true 的智能体注入合作倾向偏差
- bias=0 时等价于原始行为
"""
function biased_action_selection(agent, opponent_label, config, bias::Float64)
    # 原始的EFE行动选择
    game = config.game_type
    G_cooperate = ArbitraryInstitutions.ActionSelection.compute_expected_free_energy(
        agent, true, opponent_label, game
    )
    G_defect = ArbitraryInstitutions.ActionSelection.compute_expected_free_energy(
        agent, false, opponent_label, game
    )

    β = agent.cognitive_state.action_precision
    policy = ArbitraryInstitutions.ActionSelection.softmax([-G_cooperate, -G_defect], β)
    p_cooperate = policy[1]

    # 注入偏差：只对 label=true 的智能体
    if agent.label
        p_cooperate = min(1.0, p_cooperate + bias)
    end

    return rand() < p_cooperate
end

"""
带偏差的单次交互
"""
function biased_interaction!(agent1, agent2, config, bias::Float64)
    # 带偏差的行动选择
    action1 = biased_action_selection(agent1, agent2.label, config, bias)
    action2 = biased_action_selection(agent2, agent1.label, config, bias)

    # 环境解析（标签盲！）
    payoff1, payoff2 = ArbitraryInstitutions.Physics.resolve_interaction(
        action1, action2, config.game_type
    )

    # 创建记录
    record1 = ArbitraryInstitutions.WorldTypes.InteractionRecord(
        opponent_label = agent2.label,
        opponent_cooperated = action2,
        own_action = action1,
        payoff = payoff1
    )
    record2 = ArbitraryInstitutions.WorldTypes.InteractionRecord(
        opponent_label = agent1.label,
        opponent_cooperated = action1,
        own_action = action2,
        payoff = payoff2
    )

    # 更新历史
    push!(agent1.interaction_history, record1)
    push!(agent2.interaction_history, record2)

    # 更新信念
    ArbitraryInstitutions.Learning.update_beliefs!(agent1, record1, config)
    ArbitraryInstitutions.Learning.update_beliefs!(agent2, record2, config)

    # 结构学习
    if length(agent1.interaction_history) >= config.structure_learning_threshold
        ArbitraryInstitutions.Learning.maybe_switch_model!(agent1, config)
    end
    if length(agent2.interaction_history) >= config.structure_learning_threshold
        ArbitraryInstitutions.Learning.maybe_switch_model!(agent2, config)
    end

    # 更新内化
    ArbitraryInstitutions.Learning.update_internalization!(agent1, record1, config)
    ArbitraryInstitutions.Learning.update_internalization!(agent2, record2, config)

    return (record1, record2)
end

"""
带偏差的模拟步骤
"""
function biased_step!(sim, bias::Float64)
    agents = collect(allagents(sim.model))
    shuffle!(agents)

    pairs = Tuple{ArbitraryInstitutions.WorldTypes.InstitutionAgent,
                  ArbitraryInstitutions.WorldTypes.InstitutionAgent}[]

    for i in 1:2:length(agents)-1
        push!(pairs, (agents[i], agents[i+1]))
    end

    if isodd(length(agents))
        push!(pairs, (agents[end], rand(agents[1:end-1])))
    end

    for (a1, a2) in pairs
        biased_interaction!(a1, a2, sim.config, bias)
    end

    sim.step_count += 1
end

"""
运行带偏差的完整模拟
"""
function run_biased_simulation(; n_agents=16, steps=200, bias=0.0, seed=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    sim = Simulation(
        n_agents = n_agents,
        game_type = PrisonersDilemma(),
        complexity_penalty = 0.05,
        seed = seed === nothing ? rand(1:10000) : seed
    )

    # 记录时间序列
    adoption_history = Float64[]
    gamma_history = Float64[]
    true_coop_history = Float64[]  # True标签组的合作率
    false_coop_history = Float64[]  # False标签组的合作率

    for step in 1:steps
        biased_step!(sim, bias)

        # 每10步记录
        if step % 10 == 0
            adoption = institutional_adoption_rate(sim)
            avg_gamma = mean_internalization(sim)
            push!(adoption_history, adoption)
            push!(gamma_history, avg_gamma)

            # 计算各组的合作率
            true_agents = filter(a -> a.label, collect(allagents(sim.model)))
            false_agents = filter(a -> !a.label, collect(allagents(sim.model)))

            true_coop = 0
            true_total = 0
            for a in true_agents
                for r in a.interaction_history[max(1,end-9):end]  # 最近10次
                    true_total += 1
                    if r.own_action
                        true_coop += 1
                    end
                end
            end

            false_coop = 0
            false_total = 0
            for a in false_agents
                for r in a.interaction_history[max(1,end-9):end]
                    false_total += 1
                    if r.own_action
                        false_coop += 1
                    end
                end
            end

            push!(true_coop_history, true_total > 0 ? true_coop / true_total : 0.0)
            push!(false_coop_history, false_total > 0 ? false_coop / false_total : 0.0)
        end
    end

    return (
        final_adoption = institutional_adoption_rate(sim),
        final_gamma = mean_internalization(sim),
        final_correlation = label_correlation(sim),
        adoption_history = adoption_history,
        gamma_history = gamma_history,
        true_coop_history = true_coop_history,
        false_coop_history = false_coop_history,
        sim = sim
    )
end

# ============================================================
# 主实验：偏差扫描
# ============================================================

println("\n### 实验1：偏差扫描 (单次运行) ###\n")

biases = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

println("Bias   | 采纳率 | 平均γ | 相关性 | True组合作 | False组合作 | 差异")
println("-" ^ 75)

for bias in biases
    result = run_biased_simulation(n_agents=16, steps=300, bias=bias, seed=42)

    # 计算最终合作率
    true_coop = result.true_coop_history[end]
    false_coop = result.false_coop_history[end]

    @printf("%.2f   | %5.1f%% | %.2f  | %+.3f  |   %5.1f%%    |   %5.1f%%    | %+5.1f%%\n",
        bias,
        result.final_adoption * 100,
        result.final_gamma,
        result.final_correlation,
        true_coop * 100,
        false_coop * 100,
        (true_coop - false_coop) * 100
    )
end

# ============================================================
# 实验2：多次重复以获得统计显著性
# ============================================================

println("\n\n### 实验2：多次重复 (10次/偏差) ###\n")

n_repeats = 10
biases_detailed = [0.0, 0.05, 0.10, 0.15, 0.20]

println("Bias   | 采纳率(mean±std) | γ(mean±std) | 相关性(mean±std)")
println("-" ^ 65)

for bias in biases_detailed
    adoptions = Float64[]
    gammas = Float64[]
    correlations = Float64[]

    for rep in 1:n_repeats
        result = run_biased_simulation(n_agents=16, steps=300, bias=bias, seed=rep*100)
        push!(adoptions, result.final_adoption)
        push!(gammas, result.final_gamma)
        push!(correlations, result.final_correlation)
    end

    @printf("%.2f   |   %4.1f%% ± %4.1f%%  | %.2f ± %.2f | %+.3f ± %.3f\n",
        bias,
        mean(adoptions) * 100, std(adoptions) * 100,
        mean(gammas), std(gammas),
        mean(correlations), std(correlations)
    )
end

# ============================================================
# 实验3：时间序列追踪（特定偏差值）
# ============================================================

println("\n\n### 实验3：时间序列追踪 (bias=0.10) ###\n")

result = run_biased_simulation(n_agents=16, steps=500, bias=0.10, seed=42)

println("Step  | 采纳率 | 平均γ | True合作 | False合作 | 差异")
println("-" ^ 60)

for (i, step) in enumerate(10:10:500)
    if i <= length(result.adoption_history)
        @printf("%4d  | %5.1f%% | %.2f  |  %5.1f%%  |  %5.1f%%   | %+5.1f%%\n",
            step,
            result.adoption_history[i] * 100,
            result.gamma_history[i],
            result.true_coop_history[i] * 100,
            result.false_coop_history[i] * 100,
            (result.true_coop_history[i] - result.false_coop_history[i]) * 100
        )
    end
end

# ============================================================
# 实验4：自我实现预言检测
# ============================================================

println("\n\n### 实验4：自我实现预言检测 ###\n")

println("""
问题：即使偏差在模拟中期被移除，制度是否能自我维持？

设计：
1. 前100步：应用bias=0.15
2. 后200步：移除偏差(bias=0.0)
3. 观察制度是否持续
""")

# 自定义模拟：分两阶段
sim = Simulation(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.05,
    seed = 42
)

println("\nStep  | 阶段     | 采纳率 | 平均γ | True合作 | False合作")
println("-" ^ 65)

for step in 1:300
    # 阶段1：有偏差
    if step <= 100
        biased_step!(sim, 0.15)
        phase = "有偏差  "
    else
        # 阶段2：无偏差
        biased_step!(sim, 0.0)
        phase = "无偏差  "
    end

    if step % 20 == 0
        adoption = institutional_adoption_rate(sim)
        avg_gamma = mean_internalization(sim)

        # 计算最近的合作率
        true_agents = filter(a -> a.label, collect(allagents(sim.model)))
        false_agents = filter(a -> !a.label, collect(allagents(sim.model)))

        true_coop = 0
        true_total = 0
        for a in true_agents
            recent = a.interaction_history[max(1,end-19):end]
            for r in recent
                true_total += 1
                if r.own_action
                    true_coop += 1
                end
            end
        end

        false_coop = 0
        false_total = 0
        for a in false_agents
            recent = a.interaction_history[max(1,end-19):end]
            for r in recent
                false_total += 1
                if r.own_action
                    false_coop += 1
                end
            end
        end

        true_rate = true_total > 0 ? true_coop / true_total : 0.0
        false_rate = false_total > 0 ? false_coop / false_total : 0.0

        @printf("%4d  | %s | %5.1f%% | %.2f  |  %5.1f%%  |  %5.1f%%\n",
            step, phase,
            adoption * 100,
            avg_gamma,
            true_rate * 100,
            false_rate * 100
        )
    end
end

println("\n" * "=" ^ 80)
println("### 结论 ###")
println("=" ^ 80)
println("""
1. 最小触发偏差：观察哪个bias值开始产生显著的制度涌现
2. 自我实现性：如果移除偏差后制度仍然持续，证明"自我实现预言"机制

关键洞见：
- 初始偏差 → 行为差异 → 智能体学习到差异 → 采用制度模型 →
  基于标签做出不同预测 → 基于预测做出不同行动 → 行为差异持续

这就是"任意制度如何从微小扰动中涌现并自我维持"的核心机制！
""")

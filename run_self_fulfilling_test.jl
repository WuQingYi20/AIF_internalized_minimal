using ArbitraryInstitutions
using Printf
using Statistics: mean, std
using Random

println("=" ^ 80)
println("自我实现预言严格测试：制度能否在移除初始偏差后自我维持？")
println("=" ^ 80)

# 复用bias_sweep中的函数
include("run_bias_sweep.jl")

println("\n\n" * "=" ^ 80)
println("### 实验5：临界质量测试 ###")
println("=" ^ 80)

println("""
设计：
- 不同的初始偏差强度 (0.15, 0.20, 0.25)
- 不同的偏差持续时间 (50, 100, 200步)
- 观察移除偏差后的长期行为 (额外300步)

关键指标：
1. 采纳率是否持续？
2. 行为差异是否持续？
""")

function test_critical_mass(bias::Float64, bias_duration::Int, post_bias_duration::Int, seed::Int)
    Random.seed!(seed)

    sim = Simulation(
        n_agents = 16,
        game_type = PrisonersDilemma(),
        complexity_penalty = 0.05,
        seed = seed
    )

    # Phase 1: 有偏差
    for step in 1:bias_duration
        biased_step!(sim, bias)
    end

    adoption_after_bias = institutional_adoption_rate(sim)
    gamma_after_bias = mean_internalization(sim)

    # Phase 2: 无偏差
    for step in 1:post_bias_duration
        biased_step!(sim, 0.0)
    end

    # 计算最终合作率差异
    true_agents = filter(a -> a.label, collect(allagents(sim.model)))
    false_agents = filter(a -> !a.label, collect(allagents(sim.model)))

    true_coop = 0
    true_total = 0
    for a in true_agents
        recent = a.interaction_history[max(1,end-49):end]
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
        recent = a.interaction_history[max(1,end-49):end]
        for r in recent
            false_total += 1
            if r.own_action
                false_coop += 1
            end
        end
    end

    true_rate = true_total > 0 ? true_coop / true_total : 0.0
    false_rate = false_total > 0 ? false_coop / false_total : 0.0

    return (
        adoption_after_bias = adoption_after_bias,
        gamma_after_bias = gamma_after_bias,
        final_adoption = institutional_adoption_rate(sim),
        final_gamma = mean_internalization(sim),
        true_coop = true_rate,
        false_coop = false_rate,
        coop_diff = true_rate - false_rate
    )
end

# 测试不同配置
println("\nBias | 持续 | 采纳(后) | γ(后) | 采纳(终) | γ(终) | True% | False% | 差异 | 维持?")
println("-" ^ 90)

configurations = [
    (0.15, 50),
    (0.15, 100),
    (0.15, 200),
    (0.20, 50),
    (0.20, 100),
    (0.20, 200),
    (0.25, 50),
    (0.25, 100),
    (0.25, 200),
]

for (bias, duration) in configurations
    result = test_critical_mass(bias, duration, 300, 42)

    # 判断是否自我维持
    maintained = result.final_adoption > 0.5 && result.coop_diff > 0.05
    status = maintained ? "✓ YES" : "✗ NO"

    @printf("%.2f | %3d  |  %5.1f%%  | %.2f  |  %5.1f%%  | %.2f  | %5.1f%% | %5.1f%% | %+5.1f%% | %s\n",
        bias,
        duration,
        result.adoption_after_bias * 100,
        result.gamma_after_bias,
        result.final_adoption * 100,
        result.final_gamma,
        result.true_coop * 100,
        result.false_coop * 100,
        result.coop_diff * 100,
        status
    )
end

# 详细时间序列追踪最佳配置
println("\n\n" * "=" ^ 80)
println("### 详细追踪: bias=0.25, 200步 ###")
println("=" ^ 80)

Random.seed!(42)
sim = Simulation(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.05,
    seed = 42
)

println("\nStep  | 阶段     | 采纳率 | 平均γ | True合作 | False合作 | 差异")
println("-" ^ 70)

for step in 1:500
    if step <= 200
        biased_step!(sim, 0.25)
        phase = "有偏差  "
    else
        biased_step!(sim, 0.0)
        phase = "无偏差  "
    end

    if step % 25 == 0
        adoption = institutional_adoption_rate(sim)
        avg_gamma = mean_internalization(sim)

        # 计算最近的合作率
        true_agents = filter(a -> a.label, collect(allagents(sim.model)))
        false_agents = filter(a -> !a.label, collect(allagents(sim.model)))

        true_coop = 0
        true_total = 0
        for a in true_agents
            recent = a.interaction_history[max(1,end-24):end]
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
            recent = a.interaction_history[max(1,end-24):end]
            for r in recent
                false_total += 1
                if r.own_action
                    false_coop += 1
                end
            end
        end

        true_rate = true_total > 0 ? true_coop / true_total : 0.0
        false_rate = false_total > 0 ? false_coop / false_total : 0.0

        marker = step == 200 ? " ← 移除偏差" : ""
        @printf("%4d  | %s | %5.1f%% | %.2f  |  %5.1f%%  |  %5.1f%%   | %+5.1f%%%s\n",
            step, phase,
            adoption * 100,
            avg_gamma,
            true_rate * 100,
            false_rate * 100,
            (true_rate - false_rate) * 100,
            marker
        )
    end
end

println("\n" * "=" ^ 80)
println("### 理论分析 ###")
println("=" ^ 80)
println("""
自我实现预言的临界条件：

1. **临界采纳率**: 需要足够多的INSTITUTIONAL智能体
   - 如果太少，非互惠者占主导，信号被稀释
   - 实验表明 ~50%+ 采纳率可能是临界点

2. **临界γ值**: 需要足够强的互惠权重
   - γ决定reciprocity项在EFE中的权重
   - 实验表明 γ > 5 时互惠效应显著

3. **信念持久性**: 贝叶斯更新的问题
   - 当真实行为差异消失，信念最终会收敛
   - 需要行为差异本身能自我维持

反馈循环：
  相信差异 → 行动差异 → 观察差异 → 维持信念
      ↑                              ↓
      └────────────────────────────←─┘

断裂点：如果 "行动差异" 不够大，"观察差异" 就消失，循环断裂。
""")

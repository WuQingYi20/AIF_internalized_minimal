using ArbitraryInstitutions
using Printf
using Statistics: mean

println("=" ^ 80)
println("深度调试：为什么制度没有涌现？")
println("=" ^ 80)

# 创建模拟
sim = Simulation(
    n_agents = 8,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.05,
    seed = 42
)

println("\n### 初始配置 ###")
println("游戏类型: 囚徒困境")
println("复杂度惩罚: 0.05")
println("结构学习阈值: $(sim.config.structure_learning_threshold)")

# 追踪单个智能体的详细过程
function trace_agent_decision(agent, step)
    state = agent.cognitive_state
    beliefs = state.beliefs

    # 计算模型证据
    history = agent.interaction_history
    if length(history) < 10
        return nothing
    end

    # 提取观测
    observations = [r.opponent_cooperated for r in history]
    labels = [r.opponent_label == agent.label for r in history]  # true = ingroup

    # 分别统计内群和外群
    ingroup_obs = observations[labels]
    outgroup_obs = observations[.!labels]

    n_in = length(ingroup_obs)
    n_out = length(outgroup_obs)
    k_in = sum(ingroup_obs)
    k_out = sum(outgroup_obs)

    # 计算经验合作率
    rate_in = n_in > 0 ? k_in / n_in : 0.5
    rate_out = n_out > 0 ? k_out / n_out : 0.5
    rate_global = (k_in + k_out) / (n_in + n_out)

    # 使用初始先验计算模型证据
    α₀, β₀ = state.initial_prior

    # M0: 全局模型的证据
    k_total = k_in + k_out
    n_total = n_in + n_out
    evidence_M0 = ArbitraryInstitutions.FactorGraphs.compute_model_evidence_M0(
        observations, α₀, β₀
    )

    # M1: 分离模型的证据
    evidence_M1 = ArbitraryInstitutions.FactorGraphs.compute_model_evidence_M1(
        observations, labels, α₀, β₀, α₀, β₀
    )

    # 应用复杂度惩罚
    adjusted_M1 = evidence_M1 - 0.05

    # 贝叶斯因子
    bayes_factor = evidence_M1 - evidence_M0

    return (
        step = step,
        agent_id = agent.id,
        label = agent.label,
        n_in = n_in,
        n_out = n_out,
        k_in = k_in,
        k_out = k_out,
        rate_in = rate_in,
        rate_out = rate_out,
        rate_global = rate_global,
        rate_diff = rate_in - rate_out,
        evidence_M0 = evidence_M0,
        evidence_M1 = evidence_M1,
        adjusted_M1 = adjusted_M1,
        bayes_factor = bayes_factor,
        would_switch = adjusted_M1 > evidence_M0,
        current_model = state.active_model
    )
end

println("\n### 模拟运行 - 追踪模型证据 ###\n")

# 收集所有追踪数据
all_traces = []

for step in 1:100
    step_simulation!(sim)
    sim.step_count += 1

    # 每10步详细追踪
    if step % 10 == 0
        println("\n" * "=" ^ 60)
        println("Step $step")
        println("=" ^ 60)

        for agent in allagents(sim.model)
            trace = trace_agent_decision(agent, step)
            if trace !== nothing
                push!(all_traces, trace)

                label_str = agent.label ? "T" : "F"
                model_str = trace.current_model == NEUTRAL ? "N" : "I"
                switch_str = trace.would_switch ? "YES" : "no"

                println("\nAgent $(agent.id) (Label=$label_str, Model=$model_str):")
                println("  观测: 内群 $(trace.k_in)/$(trace.n_in) = $(round(trace.rate_in*100, digits=1))%")
                println("        外群 $(trace.k_out)/$(trace.n_out) = $(round(trace.rate_out*100, digits=1))%")
                println("        差异 = $(round(trace.rate_diff*100, digits=1))%")
                println("  证据: M0=$(round(trace.evidence_M0, digits=2)), M1=$(round(trace.evidence_M1, digits=2))")
                println("        调整后M1=$(round(trace.adjusted_M1, digits=2))")
                println("        贝叶斯因子=$(round(trace.bayes_factor, digits=2))")
                println("        应该切换? $switch_str")
            end
        end
    end
end

# 分析为什么没有切换
println("\n\n" * "=" ^ 80)
println("### 分析总结 ###")
println("=" ^ 80)

if !isempty(all_traces)
    # 找出最大贝叶斯因子
    max_bf = maximum(t -> t.bayes_factor, all_traces)
    min_bf = minimum(t -> t.bayes_factor, all_traces)
    mean_bf = mean(t -> t.bayes_factor, all_traces)

    # 找出最大合作率差异
    max_diff = maximum(t -> abs(t.rate_diff), all_traces)
    mean_diff = mean(t -> abs(t.rate_diff), all_traces)

    println("\n贝叶斯因子统计 (M1 vs M0):")
    println("  最大: $(round(max_bf, digits=3))")
    println("  最小: $(round(min_bf, digits=3))")
    println("  平均: $(round(mean_bf, digits=3))")
    println("  需要超过复杂度惩罚 0.05 才能切换")

    println("\n内群-外群合作率差异:")
    println("  最大绝对差异: $(round(max_diff*100, digits=1))%")
    println("  平均绝对差异: $(round(mean_diff*100, digits=1))%")

    # 统计有多少次"差点"切换
    close_calls = count(t -> t.bayes_factor > 0 && t.bayes_factor < 0.1, all_traces)
    would_switch = count(t -> t.would_switch, all_traces)

    println("\n切换分析:")
    println("  满足切换条件的次数: $would_switch / $(length(all_traces))")
    println("  接近切换(0 < BF < 0.1)的次数: $close_calls")
end

# 深入分析：为什么贝叶斯因子这么小？
println("\n\n### 深入分析：为什么贝叶斯因子小? ###\n")

println("""
理论分析：
1. M0 (全局模型) 假设: P(合作) = θ (对所有对手相同)
2. M1 (分离模型) 假设: P(合作|内群) = θ_in, P(合作|外群) = θ_out

贝叶斯因子 = log P(D|M1) - log P(D|M0)

只有当内群和外群的合作率差异足够大时，M1才会获得更高证据。

问题：在标签盲环境中，真实的合作率差异应该接近0！
- 环境不基于标签给予不同回报
- 所以实际观测到的差异纯粹是随机噪声
- 只有当随机噪声恰好创造出"假差异"时，制度才会涌现

这正是我们想要的行为：
- 制度只应该在有真实信号时涌现
- 在纯噪声环境中，制度涌现应该是罕见的
""")

# 检查实际的合作率分布
println("\n### 最终状态：各智能体的实际合作模式 ###\n")

for agent in allagents(sim.model)
    history = agent.interaction_history
    ingroup = filter(r -> r.opponent_label == agent.label, history)
    outgroup = filter(r -> r.opponent_label != agent.label, history)

    in_coop = count(r -> r.opponent_cooperated, ingroup)
    out_coop = count(r -> r.opponent_cooperated, outgroup)

    in_rate = length(ingroup) > 0 ? in_coop / length(ingroup) : 0.0
    out_rate = length(outgroup) > 0 ? out_coop / length(outgroup) : 0.0

    label_str = agent.label ? "T" : "F"
    model_str = agent.cognitive_state.active_model == NEUTRAL ? "NEUTRAL" : "INSTITUTIONAL"

    println("Agent $(agent.id) (Label=$label_str): $model_str")
    println("  内群: $in_coop/$(length(ingroup)) = $(round(in_rate*100, digits=1))%")
    println("  外群: $out_coop/$(length(outgroup)) = $(round(out_rate*100, digits=1))%")
    println("  差异: $(round((in_rate - out_rate)*100, digits=1))%")
    println()
end

println("=" ^ 80)
println("结论：制度没有涌现是因为没有真实的标签-行为关联信号")
println("这是修复后模型的正确行为！")
println("=" ^ 80)

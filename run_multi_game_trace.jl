using ArbitraryInstitutions
using Printf
using Statistics: mean

println("=" ^ 80)
println("多博弈类型详细追踪：制度涌现的中间过程对比")
println("=" ^ 80)

# 定义追踪函数
function trace_simulation(game_type, game_name::String; n_agents=8, steps=100, seed=42)
    println("\n" * "=" ^ 80)
    println("### $game_name ###")
    println("=" ^ 80)

    sim = Simulation(
        n_agents = n_agents,
        game_type = game_type,
        complexity_penalty = 0.05,
        seed = seed
    )

    println("\n初始状态:")
    println("Agent | Label | Model | γ    | α_in/β_in | α_out/β_out")
    println("-" ^ 60)
    for agent in allagents(sim.model)
        b = agent.cognitive_state.beliefs
        model_str = agent.cognitive_state.active_model == NEUTRAL ? "N" : "I"
        @printf("  %d   |   %s   |   %s   | %.2f |  %.1f/%.1f   |   %.1f/%.1f\n",
            agent.id,
            agent.label ? "T" : "F",
            model_str,
            agent.cognitive_state.γ,
            b.α_ingroup, b.β_ingroup,
            b.α_outgroup, b.β_outgroup
        )
    end

    # 记录关键时刻
    key_events = String[]
    prev_adoption = 0

    for step in 1:steps
        step_simulation!(sim)
        sim.step_count += 1

        # 检测模型切换事件
        n_institutional = count(a -> a.cognitive_state.active_model == INSTITUTIONAL, allagents(sim.model))
        if n_institutional != prev_adoption
            push!(key_events, "Step $step: 制度采纳 $prev_adoption → $n_institutional")
            prev_adoption = n_institutional
        end

        # 每20步输出状态
        if step % 20 == 0
            avg_γ = mean(a.cognitive_state.γ for a in allagents(sim.model))

            println("\n--- Step $step ---")
            println("制度采纳: $n_institutional/$n_agents agents, 平均γ: $(round(avg_γ, digits=2))")
            println()
            println("Agent | Label | Model | γ    | 预测in | 预测out | 内群信念 | 外群信念 | 差异")
            println("-" ^ 80)

            for agent in allagents(sim.model)
                b = agent.cognitive_state.beliefs
                state = agent.cognitive_state
                model_str = state.active_model == NEUTRAL ? "N" : "I"

                pred_in = ArbitraryInstitutions.BrainTypes.predict_cooperation(state, true)
                pred_out = ArbitraryInstitutions.BrainTypes.predict_cooperation(state, false)

                in_rate = b.α_ingroup / (b.α_ingroup + b.β_ingroup)
                out_rate = b.α_outgroup / (b.α_outgroup + b.β_outgroup)
                diff = in_rate - out_rate

                @printf("  %d   |   %s   |   %s   | %.2f |  %.2f   |  %.2f   |   %.2f    |   %.2f    | %+.2f\n",
                    agent.id,
                    agent.label ? "T" : "F",
                    model_str,
                    state.γ,
                    pred_in,
                    pred_out,
                    in_rate,
                    out_rate,
                    diff
                )
            end
        end
    end

    # 关键事件总结
    if !isempty(key_events)
        println("\n关键事件:")
        for event in key_events
            println("  • $event")
        end
    end

    # 最终交互模式分析
    println("\n交互模式分析:")
    total_in_coop = 0
    total_in_defect = 0
    total_out_coop = 0
    total_out_defect = 0

    for agent in allagents(sim.model)
        history = agent.interaction_history
        ingroup_hist = filter(r -> r.opponent_label == agent.label, history)
        outgroup_hist = filter(r -> r.opponent_label != agent.label, history)

        total_in_coop += count(r -> r.opponent_cooperated, ingroup_hist)
        total_in_defect += count(r -> !r.opponent_cooperated, ingroup_hist)
        total_out_coop += count(r -> r.opponent_cooperated, outgroup_hist)
        total_out_defect += count(r -> !r.opponent_cooperated, outgroup_hist)
    end

    in_total = total_in_coop + total_in_defect
    out_total = total_out_coop + total_out_defect

    in_rate = in_total > 0 ? total_in_coop / in_total : 0.0
    out_rate = out_total > 0 ? total_out_coop / out_total : 0.0

    println("  内群合作率: $(round(in_rate*100, digits=1))% ($total_in_coop/$in_total)")
    println("  外群合作率: $(round(out_rate*100, digits=1))% ($total_out_coop/$out_total)")
    println("  差异: $(round((in_rate - out_rate)*100, digits=1))%")

    # 返回最终指标
    return (
        adoption = institutional_adoption_rate(sim),
        gamma = mean_internalization(sim),
        correlation = label_correlation(sim),
        in_coop = in_rate,
        out_coop = out_rate
    )
end

# 运行三种博弈类型
results = Dict{String, NamedTuple}()

results["囚徒困境"] = trace_simulation(PrisonersDilemma(), "囚徒困境 (Prisoner's Dilemma)")
results["猎鹿博弈"] = trace_simulation(StagHunt(), "猎鹿博弈 (Stag Hunt)")
results["和谐博弈"] = trace_simulation(Harmony(), "和谐博弈 (Harmony)")

# 对比总结
println("\n" * "=" ^ 80)
println("### 三种博弈类型对比总结 ###")
println("=" ^ 80)

println("\n博弈类型      | 制度采纳率 | 平均γ | 标签相关性 | 内群合作 | 外群合作 | 合作差异")
println("-" ^ 90)

for (name, r) in [("囚徒困境", results["囚徒困境"]),
                   ("猎鹿博弈", results["猎鹿博弈"]),
                   ("和谐博弈", results["和谐博弈"])]
    @printf("%-12s |   %5.1f%%   | %.2f  |   %+.3f   |  %5.1f%%  |  %5.1f%%  |  %+5.1f%%\n",
        name,
        r.adoption * 100,
        r.gamma,
        r.correlation,
        r.in_coop * 100,
        r.out_coop * 100,
        (r.in_coop - r.out_coop) * 100
    )
end

println("\n" * "=" ^ 80)
println("分析解读:")
println("=" ^ 80)
println("""
1. 囚徒困境：
   - 存在合作与背叛的张力，行为有变异性
   - 智能体能学习到内群/外群的行为差异
   - 制度可以涌现，但需要一定的"幸运"初始经验

2. 猎鹿博弈：
   - 悲观先验导致快速收敛到安全策略（都背叛）
   - 行为高度一致，几乎没有变异性
   - 标签不提供任何预测价值 → 制度难以涌现

3. 和谐博弈：
   - 合作是优势策略，但仍存在随机探索
   - 合作率高但非100%，保留了可学习的变异性
   - 智能体可能发现伪相关 → 制度更容易涌现

核心洞见：制度涌现 = f(行为变异性, 标签-行为关联的可学习性)
- 太少变异 → 没有pattern可学
- 太多噪声 → pattern不稳定
- "恰到好处"的变异性才能支持制度涌现
""")

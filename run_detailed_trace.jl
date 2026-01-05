using ArbitraryInstitutions
using Printf
using Statistics: mean

println("=" ^ 70)
println("详细追踪：制度涌现的中间过程")
println("=" ^ 70)

# 创建一个小规模模拟便于观察
sim = Simulation(
    n_agents = 8,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.05,
    seed = 42
)

println("\n### 初始状态 ###\n")
println("Agent | Label | Model      | γ    | α_in/β_in | α_out/β_out | α_g/β_g")
println("-" ^ 70)
for agent in allagents(sim.model)
    b = agent.cognitive_state.beliefs
    model_str = agent.cognitive_state.active_model == NEUTRAL ? "NEUTRAL" : "INSTITUTIONAL"
    @printf("  %d   |   %s   | %-11s | %.2f |  %.1f/%.1f   |   %.1f/%.1f    | %.1f/%.1f\n",
        agent.id,
        agent.label ? "T" : "F",
        model_str,
        agent.cognitive_state.γ,
        b.α_ingroup, b.β_ingroup,
        b.α_outgroup, b.β_outgroup,
        b.α_global, b.β_global
    )
end

# 运行并追踪
println("\n### 逐步追踪 (每10步) ###\n")

for step in 1:100
    # 执行一步
    step_simulation!(sim)
    sim.step_count += 1

    # 每10步输出详细状态
    if step % 10 == 0
        println("\n--- Step $step ---")

        # 统计
        n_institutional = count(a -> a.cognitive_state.active_model == INSTITUTIONAL, allagents(sim.model))
        avg_γ = mean(a.cognitive_state.γ for a in allagents(sim.model))

        println("制度采纳: $n_institutional/8 agents, 平均γ: $(round(avg_γ, digits=2))")
        println()
        println("Agent | Label | Model | γ    | 预测in | 预测out | 交互数 | 最近对手 | 对手合作?")
        println("-" ^ 75)

        for agent in allagents(sim.model)
            b = agent.cognitive_state.beliefs
            state = agent.cognitive_state
            model_str = state.active_model == NEUTRAL ? "N" : "I"

            # 计算预测的合作率
            pred_in = ArbitraryInstitutions.BrainTypes.predict_cooperation(state, true)
            pred_out = ArbitraryInstitutions.BrainTypes.predict_cooperation(state, false)

            # 最近交互
            n_hist = length(agent.interaction_history)
            if n_hist > 0
                last = agent.interaction_history[end]
                last_opp = last.opponent_label ? "T" : "F"
                last_coop = last.opponent_cooperated ? "Y" : "N"
            else
                last_opp = "-"
                last_coop = "-"
            end

            @printf("  %d   |   %s   |   %s   | %.2f |  %.2f   |  %.2f   |   %2d   |    %s     |     %s\n",
                agent.id,
                agent.label ? "T" : "F",
                model_str,
                state.γ,
                pred_in,
                pred_out,
                n_hist,
                last_opp,
                last_coop
            )
        end

        # 模型切换事件检测
        if step >= 20
            println()
            println("信念状态详情:")
            for agent in allagents(sim.model)
                b = agent.cognitive_state.beliefs
                in_rate = b.α_ingroup / (b.α_ingroup + b.β_ingroup)
                out_rate = b.α_outgroup / (b.α_outgroup + b.β_outgroup)
                global_rate = b.α_global / (b.α_global + b.β_global)

                @printf("  Agent %d: 全局=%.2f, 内群=%.2f, 外群=%.2f, 差异=%.2f\n",
                    agent.id, global_rate, in_rate, out_rate, in_rate - out_rate)
            end
        end
    end
end

# 最终详细分析
println("\n" * "=" ^ 70)
println("### 最终状态分析 (100步后) ###")
println("=" ^ 70)

println("\n各智能体详细状态:")
println("Agent | Label | Model | γ    | 全局信念 | 内群信念 | 外群信念 | 内外差")
println("-" ^ 70)

for agent in allagents(sim.model)
    b = agent.cognitive_state.beliefs
    state = agent.cognitive_state
    model_str = state.active_model == NEUTRAL ? "NEUTRAL" : "INSTIT."

    in_rate = b.α_ingroup / (b.α_ingroup + b.β_ingroup)
    out_rate = b.α_outgroup / (b.α_outgroup + b.β_outgroup)
    global_rate = b.α_global / (b.α_global + b.β_global)
    diff = in_rate - out_rate

    @printf("  %d   |   %s   | %-7s | %.2f |   %.3f   |   %.3f   |   %.3f   | %+.3f\n",
        agent.id,
        agent.label ? "T" : "F",
        model_str,
        state.γ,
        global_rate,
        in_rate,
        out_rate,
        diff
    )
end

# 交互模式分析
println("\n交互模式分析:")
for agent in allagents(sim.model)
    history = agent.interaction_history
    n_total = length(history)

    # 分组统计
    ingroup_hist = filter(r -> r.opponent_label == agent.label, history)
    outgroup_hist = filter(r -> r.opponent_label != agent.label, history)

    in_coop_received = count(r -> r.opponent_cooperated, ingroup_hist)
    out_coop_received = count(r -> r.opponent_cooperated, outgroup_hist)

    in_coop_given = count(r -> r.own_action, ingroup_hist)
    out_coop_given = count(r -> r.own_action, outgroup_hist)

    @printf("Agent %d (Label=%s): 内群交互%d次(收到合作%d,给出合作%d), 外群交互%d次(收到合作%d,给出合作%d)\n",
        agent.id,
        agent.label ? "T" : "F",
        length(ingroup_hist), in_coop_received, in_coop_given,
        length(outgroup_hist), out_coop_received, out_coop_given
    )
end

println("\n总体指标:")
println("  制度采纳率: $(round(institutional_adoption_rate(sim)*100, digits=1))%")
println("  平均内化度: $(round(mean_internalization(sim), digits=2))")
println("  标签相关性: $(round(label_correlation(sim), digits=3))")

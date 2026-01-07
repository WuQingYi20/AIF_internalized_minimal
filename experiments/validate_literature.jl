"""
    Literature Validation: Testing Model Predictions Against Empirical Findings

验证模型预测是否匹配实证文献:
1. Yamagishi (1999): 高信任者检测准确性
2. Balliet (2014): 效应量 d = 0.32
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")
using Printf
using Statistics

# ============================================================
# 验证1: 效应量校准 (Balliet et al., 2014)
# 目标: d = 0.32 (合作差异)
# ============================================================

function validate_effect_size()
    println("=" ^ 80)
    println("验证1: 效应量校准 (Balliet et al., 2014 报告 d = 0.32)")
    println("=" ^ 80)

    # 运行多次模拟，计算 Cohen's d
    configs = ExperimentConfig[]

    # 使用最优参数
    for rep in 1:100
        seed = 50000 + rep
        config = default_config(
            experiment_id = "effect_size",
            config_id = 1,
            seed = seed,
            bias = 0.10,
            bias_duration = 100,
            post_bias_duration = 300,
            action_precision = 5.0,  # 最优
            complexity_penalty = 0.05  # 最优
        )
        push!(configs, config)
    end

    df = run_experiment_batch(configs)

    # 计算 Cohen's d
    # d = (M_true - M_false) / SD_pooled
    true_coop = df.true_cooperation_rate
    false_coop = df.false_cooperation_rate
    gaps = true_coop .- false_coop

    mean_gap = mean(gaps)
    sd_pooled = sqrt((var(true_coop) + var(false_coop)) / 2)
    cohens_d = mean_gap / sd_pooled

    println("\n合作率统计:")
    println("  True组平均: $(round(mean(true_coop)*100, digits=1))%")
    println("  False组平均: $(round(mean(false_coop)*100, digits=1))%")
    println("  平均差异: $(round(mean_gap*100, digits=1))%")
    println("  混合标准差: $(round(sd_pooled*100, digits=1))%")
    println("\nCohen's d = $(round(cohens_d, digits=2))")
    println("Balliet (2014) 报告: d = 0.32")

    if abs(cohens_d - 0.32) < 0.15
        println("✅ 效应量与元分析一致!")
    else
        println("⚠ 效应量有差异，需要调整参数")
    end

    return cohens_d
end

# ============================================================
# 验证2: 高信任者欺骗检测 (Yamagishi, 1999)
# "高信任者在匿名条件下更准确检测欺骗者"
# ============================================================

function validate_yamagishi_detection()
    println("\n" * "=" ^ 80)
    println("验证2: 高信任者检测准确性 (Yamagishi, 1999)")
    println("假说: 高信任者虽然初始更合作，但检测差异更准确")
    println("=" ^ 80)

    priors = [
        ((1.0, 3.0), "低信任 E[p]=0.25"),
        ((1.0, 1.0), "中信任 E[p]=0.50"),
        ((3.0, 1.0), "高信任 E[p]=0.75"),
    ]

    results = []

    for (prior, name) in priors
        configs = ExperimentConfig[]

        for rep in 1:50
            seed = 60000 + rep
            config = default_config(
                experiment_id = "detection",
                config_id = 1,
                seed = seed,
                prior_cooperation = prior,
                bias = 0.10,
                bias_duration = 100,
                post_bias_duration = 300
            )
            push!(configs, config)
        end

        df = run_experiment_batch(configs, show_progress=false)

        # 检测准确性 = 信念差异是否正确反映行为差异
        # 真实行为差异 > 0 (True组更合作)
        # 检测准确 = belief_difference > 0 的比例
        accurate_detection = mean(df.belief_difference .> 0)

        # 信念更新幅度
        belief_change = mean(abs.(df.belief_difference))

        push!(results, (
            name = name,
            prior = prior,
            mean_coop_gap = mean(df.cooperation_gap),
            belief_diff = mean(df.belief_difference),
            detection_rate = accurate_detection,
            belief_change = belief_change
        ))
    end

    println("\n信任水平 | 合作差 | 信念差 | 检测率 | 信念变化幅度")
    println("-" ^ 70)

    for r in results
        @printf("%-20s | %+5.1f%% | %+.3f | %5.1f%% | %.4f\n",
            r.name,
            r.mean_coop_gap * 100,
            r.belief_diff,
            r.detection_rate * 100,
            r.belief_change
        )
    end

    # Yamagishi预测: 高信任者检测更准确
    low_trust_detection = results[1].detection_rate
    high_trust_detection = results[3].detection_rate

    println("\nYamagishi (1999) 预测验证:")
    println("  低信任者检测率: $(round(low_trust_detection*100, digits=1))%")
    println("  高信任者检测率: $(round(high_trust_detection*100, digits=1))%")

    if high_trust_detection > low_trust_detection
        println("  ✅ 高信任者检测更准确 - 与Yamagishi一致!")
    else
        println("  ❌ 与Yamagishi预测不符")
        println("  但注意: Yamagishi发现是在*匿名*条件下")
        println("  我们模型可能需要添加匿名/非匿名机制")
    end

    return results
end

# ============================================================
# 验证3: 内群偏好 vs 外群贬低 (Balliet, 2014)
# ============================================================

function validate_ingroup_favoritism()
    println("\n" * "=" ^ 80)
    println("验证3: 内群偏好 vs 外群贬低 (Balliet, 2014)")
    println("元分析发现: 主要是内群偏好，而非外群贬低")
    println("=" ^ 80)

    configs = ExperimentConfig[]

    for rep in 1:100
        seed = 70000 + rep
        config = default_config(
            experiment_id = "favoritism",
            config_id = 1,
            seed = seed,
            bias = 0.10,
            bias_duration = 100,
            post_bias_duration = 300
        )
        push!(configs, config)
    end

    df = run_experiment_batch(configs)

    # 分析: 内群信念提升 vs 外群信念下降
    # 初始先验 = 0.5 (neutral)
    baseline = 0.5

    ingroup_change = mean(df.mean_ingroup_belief) - baseline
    outgroup_change = mean(df.mean_outgroup_belief) - baseline

    println("\n信念变化 (相对于基线0.5):")
    println("  内群信念变化: $(round(ingroup_change, digits=3)) ($(ingroup_change > 0 ? "提升" : "下降"))")
    println("  外群信念变化: $(round(outgroup_change, digits=3)) ($(outgroup_change > 0 ? "提升" : "下降"))")

    # 判断主导机制
    if abs(ingroup_change) > abs(outgroup_change) && ingroup_change > 0
        println("\n✅ 内群偏好主导 - 与Balliet (2014)一致!")
        println("   机制: 主要是对内群的积极信念提升")
    elseif abs(outgroup_change) > abs(ingroup_change) && outgroup_change < 0
        println("\n外群贬低主导 - 与Balliet (2014)不一致")
    else
        println("\n混合机制或无显著偏好")
    end

    return (ingroup_change, outgroup_change)
end

# ============================================================
# 主函数
# ============================================================

function run_all_validations()
    println("=" ^ 80)
    println("文献验证实验")
    println("=" ^ 80)

    d = validate_effect_size()
    detection_results = validate_yamagishi_detection()
    favoritism = validate_ingroup_favoritism()

    println("\n" * "=" ^ 80)
    println("验证总结")
    println("=" ^ 80)

    println("""

    1. 效应量校准 (Balliet et al., 2014):
       - 模型 Cohen's d = $(round(d, digits=2))
       - 文献 Cohen's d = 0.32
       - 状态: $(abs(d - 0.32) < 0.15 ? "✅ 匹配" : "⚠ 需调整")

    2. 高信任者检测 (Yamagishi, 1999):
       - 预测: 高信任者检测更准确
       - 结果: $(detection_results[3].detection_rate > detection_results[1].detection_rate ? "✅ 一致" : "❌ 不一致")

    3. 内群偏好 vs 外群贬低 (Balliet, 2014):
       - 预测: 内群偏好主导
       - 结果: $(favoritism[1] > abs(favoritism[2]) ? "✅ 一致" : "❌ 不一致")

    """)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_all_validations()
end

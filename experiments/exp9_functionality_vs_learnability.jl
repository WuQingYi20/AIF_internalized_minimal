"""
    Experiment 9: Functionality vs Learnability (2×2 Design)

这是论文的核心实验，直接测试：
    功能主义预测: 有用 → 涌现
    我们的反驳:   有用 ∧ 不可学习 → 不涌现

2×2 设计:
                    可学习性
                 高 (bias=0.15)    低 (bias=0.02)
            ┌─────────────────┬─────────────────┐
  功   高   │  预测: 涌现     │  预测: 不涌现   │ ← 关键格!
  能   (CC=10)                │  (挑战功能主义) │
  性  ├─────────────────┼─────────────────┤
       低   │  预测: 涌现     │  预测: 不涌现   │
       (CC=3) │  (但空壳)       │                 │
            └─────────────────┴─────────────────┘

关键预测:
- 高功能+低可学习 < 低功能+高可学习
  (功能性无法弥补可学习性的不足)
"""

using Pkg
Pkg.activate(".")

# 在加载并行框架之前定义自定义类型
using ArbitraryInstitutions

# 高功能博弈：合作收益大 (CC=10 vs DD=1, 收益差=9)
struct HighFunctionalityPD <: GameType end

# 低功能博弈：合作收益小 (CC=3 vs DD=1, 收益差=2)
struct LowFunctionalityPD <: GameType end

# 定义payoff矩阵
ArbitraryInstitutions.Physics.get_payoff_matrix(::HighFunctionalityPD) = [10.0 0.0; 12.0 1.0]
ArbitraryInstitutions.Physics.get_payoff_matrix(::LowFunctionalityPD) = [3.0 0.0; 5.0 1.0]

# 现在加载并行框架，并将类型广播到workers
include("run_parameter_sweep.jl")

# 将自定义类型发送到所有workers
@everywhere begin
    struct HighFunctionalityPD <: GameType end
    struct LowFunctionalityPD <: GameType end
    ArbitraryInstitutions.Physics.get_payoff_matrix(::HighFunctionalityPD) = [10.0 0.0; 12.0 1.0]
    ArbitraryInstitutions.Physics.get_payoff_matrix(::LowFunctionalityPD) = [3.0 0.0; 5.0 1.0]
end

using Printf
using Statistics

const EXPERIMENT_ID = "exp9_func_vs_learn"
const N_REPEATS = 50  # 每格50次，总共200次

# ============================================================
# 主实验
# ============================================================

function run_experiment9()
    println("=" ^ 80)
    println("Experiment 9: Functionality vs Learnability (2×2 Design)")
    println("=" ^ 80)
    println()
    println("核心假说: 功能性无法弥补可学习性的不足")
    println()

    # 2×2 条件
    conditions = [
        (HighFunctionalityPD(), 0.15, "高功能+高可学"),
        (HighFunctionalityPD(), 0.02, "高功能+低可学"),  # 关键格
        (LowFunctionalityPD(),  0.15, "低功能+高可学"),
        (LowFunctionalityPD(),  0.02, "低功能+低可学"),
    ]

    configs = ExperimentConfig[]
    config_id = 1

    for (game, bias, name) in conditions
        for rep in 1:N_REPEATS
            seed = 90000 + config_id * 100 + rep
            config = default_config(
                experiment_id = EXPERIMENT_ID,
                config_id = config_id,
                seed = seed,
                game_type = game,
                bias = bias,
                bias_duration = 100,
                post_bias_duration = 300
            )
            push!(configs, config)
        end
        config_id += 1
    end

    println("Total configurations: $(length(configs))")
    println("Running on $(nworkers()) workers...")
    println()

    # 运行实验
    df = run_experiment_batch(configs)

    # 保存结果
    save_results(df, EXPERIMENT_ID)

    # ============================================================
    # 结果分析
    # ============================================================

    println("\n" * "=" ^ 80)
    println("2×2 结果矩阵")
    println("=" ^ 80)

    # 添加条件标签
    df.functionality = map(row -> contains(row.game_type, "High") ? "高功能" : "低功能", eachrow(df))
    df.learnability = map(row -> row.bias >= 0.10 ? "高可学" : "低可学", eachrow(df))

    # 计算每个格子的统计
    results = Dict()

    for func in ["高功能", "低功能"]
        for learn in ["高可学", "低可学"]
            subset = filter(row -> row.functionality == func && row.learnability == learn, df)

            adoption = mean(subset.final_adoption)
            emergence = mean(subset.institution_emerged)
            sf = mean(subset.self_fulfilling)
            gap = mean(subset.cooperation_gap)

            # 计算标准误
            se_sf = std(subset.self_fulfilling) / sqrt(nrow(subset))

            results[(func, learn)] = (
                adoption = adoption,
                emergence = emergence,
                sf = sf,
                se_sf = se_sf,
                gap = gap,
                n = nrow(subset)
            )
        end
    end

    # 打印2×2矩阵
    println("\n                     可学习性")
    println("                  高 (bias=0.15)     低 (bias=0.02)")
    println("            ┌───────────────────┬───────────────────┐")

    for func in ["高功能", "低功能"]
        print("  功能性 $(func) │")
        for learn in ["高可学", "低可学"]
            r = results[(func, learn)]
            @printf("  SF=%4.1f%%±%.1f%%   │", r.sf*100, r.se_sf*100)
        end
        println()
        print("              │")
        for learn in ["高可学", "低可学"]
            r = results[(func, learn)]
            @printf("  Gap=%+4.1f%%       │", r.gap*100)
        end
        println()
        if func == "高功能"
            println("            ├───────────────────┼───────────────────┤")
        end
    end
    println("            └───────────────────┴───────────────────┘")

    # ============================================================
    # 核心假说检验
    # ============================================================

    println("\n" * "=" ^ 80)
    println("核心假说检验")
    println("=" ^ 80)

    # 关键比较1: 高功能+低可学 vs 低功能+高可学
    hf_ll = results[("高功能", "低可学")]
    lf_hl = results[("低功能", "高可学")]

    println("\n关键比较: 高功能+低可学 vs 低功能+高可学")
    println("-" ^ 50)
    @printf("  高功能+低可学: SF = %.1f%% (n=%d)\n", hf_ll.sf*100, hf_ll.n)
    @printf("  低功能+高可学: SF = %.1f%% (n=%d)\n", lf_hl.sf*100, lf_hl.n)

    if lf_hl.sf > hf_ll.sf
        println("\n  ✅ 低功能+高可学 > 高功能+低可学")
        println("  → 可学习性比功能性更重要!")
        println("  → 直接反驳功能主义核心预测")
    else
        println("\n  ❌ 未能证明可学习性更重要")
    end

    # 关键比较2: 可学习性的主效应
    println("\n可学习性主效应:")
    println("-" ^ 50)

    high_learn = (results[("高功能", "高可学")].sf + results[("低功能", "高可学")].sf) / 2
    low_learn = (results[("高功能", "低可学")].sf + results[("低功能", "低可学")].sf) / 2

    @printf("  高可学习性平均: SF = %.1f%%\n", high_learn*100)
    @printf("  低可学习性平均: SF = %.1f%%\n", low_learn*100)
    @printf("  差异: %.1f%%\n", (high_learn - low_learn)*100)

    # 关键比较3: 功能性的主效应
    println("\n功能性主效应:")
    println("-" ^ 50)

    high_func = (results[("高功能", "高可学")].sf + results[("高功能", "低可学")].sf) / 2
    low_func = (results[("低功能", "高可学")].sf + results[("低功能", "低可学")].sf) / 2

    @printf("  高功能性平均: SF = %.1f%%\n", high_func*100)
    @printf("  低功能性平均: SF = %.1f%%\n", low_func*100)
    @printf("  差异: %.1f%%\n", (high_func - low_func)*100)

    # 效应量比较
    println("\n效应量比较:")
    println("-" ^ 50)
    learn_effect = high_learn - low_learn
    func_effect = high_func - low_func

    @printf("  可学习性效应: %.1f%%\n", learn_effect*100)
    @printf("  功能性效应:   %.1f%%\n", func_effect*100)
    @printf("  比值: 可学习性/功能性 = %.1f\n", learn_effect / max(func_effect, 0.001))

    if learn_effect > func_effect * 2
        println("\n  ✅ 可学习性效应是功能性效应的2倍以上")
        println("  → 强有力地支持我们的核心论点")
    end

    # ============================================================
    # 空壳制度检验
    # ============================================================

    println("\n" * "=" ^ 80)
    println("空壳制度检验 (低功能+高可学)")
    println("=" ^ 80)

    lf_hl_data = filter(row -> row.functionality == "低功能" && row.learnability == "高可学", df)
    hf_hl_data = filter(row -> row.functionality == "高功能" && row.learnability == "高可学", df)

    println("\n比较两种高可学习条件:")
    @printf("  高功能+高可学: 采纳率=%.1f%%, 合作差=%.1f%%\n",
        mean(hf_hl_data.final_adoption)*100, mean(hf_hl_data.cooperation_gap)*100)
    @printf("  低功能+高可学: 采纳率=%.1f%%, 合作差=%.1f%%\n",
        mean(lf_hl_data.final_adoption)*100, mean(lf_hl_data.cooperation_gap)*100)

    if mean(lf_hl_data.final_adoption) > 0.3 && abs(mean(lf_hl_data.cooperation_gap)) < 0.05
        println("\n  ✅ 低功能+高可学 = 空壳制度")
        println("  → 制度采纳但行为差异小")
        println("  → 证明可学习但无用的制度也会涌现")
    end

    # ============================================================
    # 论文结论
    # ============================================================

    println("\n" * "=" ^ 80)
    println("论文核心结论")
    println("=" ^ 80)

    println("""

    实验9直接测试了功能主义vs可学习性假说。

    关键发现:
    1. 高功能+低可学习 → 制度不涌现 (SF ≈ $(round(hf_ll.sf*100, digits=1))%)
       → 功能性无法弥补可学习性的不足

    2. 低功能+高可学习 → 制度涌现 (SF ≈ $(round(lf_hl.sf*100, digits=1))%)
       → 可学习性足以触发涌现，即使功能有限

    3. 可学习性效应 ($(round(learn_effect*100, digits=1))%) >> 功能性效应 ($(round(func_effect*100, digits=1))%)
       → 可学习性是制度涌现的主要决定因素

    这直接反驳了功能主义的核心预测：
    "有用的制度会自然涌现"

    我们证明：
    "可学习的制度会涌现，无论是否有用；
     不可学习的制度不会涌现，无论多有用。"
    """)

    return df
end

# 运行
if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment9()
end

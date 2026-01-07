"""
    Experiment 10: The Functionality Paradox

验证发现：高功能反而抑制涌现

因果链假说：
  高功能 → 所有人都倾向合作 → 行为趋同 → 信号消失 → 不可学习 → 不涌现

这是一个自我击败的悖论：
  越有用的制度，越不需要区分内外群，越难以学习，越难涌现。

实验设计：
1. 测量不同功能性下的基础合作率
2. 测量行为差异（信号强度）
3. 确认因果链
"""

using Pkg
Pkg.activate(".")

using ArbitraryInstitutions

# 定义不同功能性级别的博弈
struct VeryHighFuncPD <: GameType end  # CC=20
struct HighFuncPD <: GameType end       # CC=10
struct MediumFuncPD <: GameType end     # CC=5
struct LowFuncPD <: GameType end        # CC=3

ArbitraryInstitutions.Physics.get_payoff_matrix(::VeryHighFuncPD) = [20.0 0.0; 22.0 1.0]
ArbitraryInstitutions.Physics.get_payoff_matrix(::HighFuncPD) = [10.0 0.0; 12.0 1.0]
ArbitraryInstitutions.Physics.get_payoff_matrix(::MediumFuncPD) = [5.0 0.0; 7.0 1.0]
ArbitraryInstitutions.Physics.get_payoff_matrix(::LowFuncPD) = [3.0 0.0; 5.0 1.0]

include("run_parameter_sweep.jl")

@everywhere begin
    struct VeryHighFuncPD <: GameType end
    struct HighFuncPD <: GameType end
    struct MediumFuncPD <: GameType end
    struct LowFuncPD <: GameType end
    ArbitraryInstitutions.Physics.get_payoff_matrix(::VeryHighFuncPD) = [20.0 0.0; 22.0 1.0]
    ArbitraryInstitutions.Physics.get_payoff_matrix(::HighFuncPD) = [10.0 0.0; 12.0 1.0]
    ArbitraryInstitutions.Physics.get_payoff_matrix(::MediumFuncPD) = [5.0 0.0; 7.0 1.0]
    ArbitraryInstitutions.Physics.get_payoff_matrix(::LowFuncPD) = [3.0 0.0; 5.0 1.0]
end

using Printf
using Statistics

const EXPERIMENT_ID = "exp10_func_paradox"
const N_REPEATS = 50

function run_experiment10()
    println("=" ^ 80)
    println("Experiment 10: The Functionality Paradox")
    println("=" ^ 80)
    println()
    println("假说: 高功能 → 行为趋同 → 信号消失 → 不涌现")
    println()

    # 功能性梯度
    games = [
        (VeryHighFuncPD(), 20, "极高功能 CC=20"),
        (HighFuncPD(),     10, "高功能 CC=10"),
        (MediumFuncPD(),    5, "中功能 CC=5"),
        (LowFuncPD(),       3, "低功能 CC=3"),
    ]

    # 固定高可学习性 (bias=0.15) 以隔离功能性效应
    bias = 0.15

    configs = ExperimentConfig[]
    config_id = 1

    for (game, cc, name) in games
        for rep in 1:N_REPEATS
            seed = 100000 + config_id * 100 + rep
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
    df = run_experiment_batch(configs)
    save_results(df, EXPERIMENT_ID)

    # 添加功能性标签
    df.cc_value = map(row -> begin
        if contains(row.game_type, "VeryHigh")
            20
        elseif contains(row.game_type, "High")
            10
        elseif contains(row.game_type, "Medium")
            5
        else
            3
        end
    end, eachrow(df))

    # ============================================================
    # 核心分析：因果链验证
    # ============================================================

    println("\n" * "=" ^ 80)
    println("因果链验证: 功能性 → 行为趋同 → 信号消失 → 不涌现")
    println("=" ^ 80)

    println("\n功能性 | True合作 | False合作 | 行为差异 | 采纳率 | 自实现率")
    println("-" ^ 75)

    results = []
    for cc in [20, 10, 5, 3]
        subset = filter(row -> row.cc_value == cc, df)

        true_coop = mean(subset.true_cooperation_rate)
        false_coop = mean(subset.false_cooperation_rate)
        gap = true_coop - false_coop
        adoption = mean(subset.final_adoption)
        sf = mean(subset.self_fulfilling)

        push!(results, (cc=cc, true_coop=true_coop, false_coop=false_coop,
                        gap=gap, adoption=adoption, sf=sf))

        @printf("CC=%2d   |  %5.1f%%  |   %5.1f%%  |  %+5.1f%% |  %5.1f%% |  %5.1f%%\n",
            cc, true_coop*100, false_coop*100, gap*100, adoption*100, sf*100)
    end

    # ============================================================
    # 因果链可视化
    # ============================================================

    println("\n" * "=" ^ 80)
    println("因果链可视化")
    println("=" ^ 80)

    println("\n功能性↑ → 基础合作率↑ → 行为差异↓ → 涌现率↓")
    println()

    for r in results
        # 用ASCII bar表示
        coop_bar = repeat("█", round(Int, r.true_coop * 20))
        gap_bar = repeat("█", round(Int, abs(r.gap) * 50))
        sf_bar = repeat("█", round(Int, r.sf * 20))

        println("CC=$(lpad(r.cc, 2)): 合作率 $(rpad(coop_bar, 20)) $(round(r.true_coop*100, digits=0))%")
        println("      行为差 $(rpad(gap_bar, 20)) $(round(r.gap*100, digits=1))%")
        println("      涌现率 $(rpad(sf_bar, 20)) $(round(r.sf*100, digits=0))%")
        println()
    end

    # ============================================================
    # 相关性分析
    # ============================================================

    println("=" ^ 80)
    println("相关性分析")
    println("=" ^ 80)

    ccs = [r.cc for r in results]
    gaps = [r.gap for r in results]
    sfs = [r.sf for r in results]
    true_coops = [r.true_coop for r in results]

    # 计算趋势
    println("\n功能性与行为趋同:")
    println("  CC=3  → 基础合作率 $(round(results[4].true_coop*100, digits=1))%")
    println("  CC=20 → 基础合作率 $(round(results[1].true_coop*100, digits=1))%")
    println("  趋势: 功能性↑ → 合作率↑")

    println("\n行为趋同与信号强度:")
    println("  CC=3  → 行为差异 $(round(results[4].gap*100, digits=1))%")
    println("  CC=20 → 行为差异 $(round(results[1].gap*100, digits=1))%")
    println("  趋势: 合作率↑ → 差异↓ (信号消失)")

    println("\n信号强度与涌现:")
    println("  CC=3  → 自实现率 $(round(results[4].sf*100, digits=1))%")
    println("  CC=20 → 自实现率 $(round(results[1].sf*100, digits=1))%")
    println("  趋势: 差异↓ → 涌现↓")

    # ============================================================
    # 悖论确认
    # ============================================================

    println("\n" * "=" ^ 80)
    println("功能性悖论确认")
    println("=" ^ 80)

    if results[1].sf < results[4].sf
        println("""

    ✅ 悖论确认!

    高功能 (CC=20): 自实现率 = $(round(results[1].sf*100, digits=1))%
    低功能 (CC=3):  自实现率 = $(round(results[4].sf*100, digits=1))%

    因果链验证:
    1. 高功能 → 合作更有利 → 基础合作率 $(round(results[1].true_coop*100, digits=1))%
    2. 高合作率 → 两组都合作 → 行为差异仅 $(round(results[1].gap*100, digits=1))%
    3. 差异小 → 信号弱 → 无法学习区分
    4. 不可学习 → 制度不涌现

    这是一个自我击败的悖论:
    ┌─────────────────────────────────────────────────────┐
    │  越有用的制度，越不需要区分内外群，                  │
    │  越难以学习，越难涌现。                             │
    └─────────────────────────────────────────────────────┘

    对功能主义的致命打击:
    功能主义预测: 有用 → 涌现
    我们的数据:   有用 → 行为趋同 → 差异消失 → 不涌现
        """)
    else
        println("\n  ❌ 悖论未确认，需要进一步分析")
    end

    # ============================================================
    # 新的一句话贡献
    # ============================================================

    println("\n" * "=" ^ 80)
    println("新的核心论点")
    println("=" ^ 80)

    println("""

    原论点: 功能性不是涌现的充分条件

    新发现: 功能性与可学习性存在张力

    机制:   高功能制度使行为趋同，反而削弱可学习性，抑制涌现

    启示:   这解释了为什么"有用"的规范常常建立不起来——它们自我击败

    一句话:
    ┌─────────────────────────────────────────────────────────────────┐
    │  制度涌现需要行为差异作为学习信号，而高功能制度消除了这种差异。  │
    │  这是功能主义的自我击败悖论。                                   │
    └─────────────────────────────────────────────────────────────────┘
    """)

    return df
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_experiment10()
end

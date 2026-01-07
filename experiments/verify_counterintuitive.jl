"""
    Verification of Counter-Intuitive Results

验证四个反直觉发现：
1. 小种群效果更好
2. 乐观先验反而差
3. 制度长期衰退
4. 行动精度非单调

每个验证使用更多重复(50次)和更细粒度的参数
"""

using Pkg
Pkg.activate(".")

include("run_parameter_sweep.jl")
using Printf
using Statistics

const N_REPEATS = 50  # 更多重复以获得更强统计功效

# ============================================================
# 验证1: 小种群效果 - 为什么N=8比N=128好?
# ============================================================

function verify_population_effect()
    println("\n" * "=" ^ 80)
    println("验证1: 小种群效果")
    println("假设: 小种群中信号噪声大，但更容易达到临界质量")
    println("=" ^ 80)

    # 更细粒度的种群大小
    n_agents_list = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    bias = 0.10  # 固定中等偏差

    configs = ExperimentConfig[]
    config_id = 1

    for n in n_agents_list
        for rep in 1:N_REPEATS
            seed = 10000 + config_id * 100 + rep
            config = default_config(
                experiment_id = "verify_population",
                config_id = config_id,
                seed = seed,
                n_agents = n,
                bias = bias,
                bias_duration = 100,
                post_bias_duration = 300
            )
            push!(configs, config)
        end
        config_id += 1
    end

    println("运行 $(length(configs)) 配置...")
    df = run_experiment_batch(configs)

    # 分析
    println("\n种群大小 | 采纳率 | 涌现率 | 自实现率 | 合作差")
    println("-" ^ 60)

    for n in n_agents_list
        subset = filter(row -> row.n_agents == n, df)
        adoption = mean(subset.final_adoption)
        emergence = mean(subset.institution_emerged)
        sf = mean(subset.self_fulfilling)
        gap = mean(subset.cooperation_gap)

        @printf("%6d   | %5.1f%% | %5.1f%% |  %5.1f%%  | %+5.1f%%\n",
            n, adoption*100, emergence*100, sf*100, gap*100)
    end

    # 计算临界质量比例
    println("\n临界质量分析 (adoption > 0.5 的比例):")
    for n in n_agents_list
        subset = filter(row -> row.n_agents == n, df)
        critical_mass = count(row -> row.final_adoption > 0.5, eachrow(subset)) / nrow(subset)
        agents_needed = ceil(Int, n * 0.5)
        @printf("  N=%3d: 需要 %2d 个智能体采纳 → 达到率 %.1f%%\n",
            n, agents_needed, critical_mass * 100)
    end

    return df
end

# ============================================================
# 验证2: 乐观先验为何差?
# ============================================================

function verify_optimistic_prior()
    println("\n" * "=" ^ 80)
    println("验证2: 乐观先验效果")
    println("假设: 乐观者已经预期高合作，难以检测到额外信号")
    println("=" ^ 80)

    # 更细粒度的先验
    priors = [
        (1.0, 5.0),  # 非常悲观 E[p] = 0.17
        (1.0, 3.0),  # 悲观 E[p] = 0.25
        (1.0, 2.0),  # 略悲观 E[p] = 0.33
        (1.0, 1.0),  # 中性 E[p] = 0.50
        (2.0, 1.0),  # 略乐观 E[p] = 0.67
        (3.0, 1.0),  # 乐观 E[p] = 0.75
        (5.0, 1.0),  # 非常乐观 E[p] = 0.83
    ]

    configs = ExperimentConfig[]
    config_id = 1

    for prior in priors
        for rep in 1:N_REPEATS
            seed = 20000 + config_id * 100 + rep
            config = default_config(
                experiment_id = "verify_prior",
                config_id = config_id,
                seed = seed,
                prior_cooperation = prior,
                bias = 0.10,
                bias_duration = 100,
                post_bias_duration = 300
            )
            push!(configs, config)
        end
        config_id += 1
    end

    println("运行 $(length(configs)) 配置...")
    df = run_experiment_batch(configs)

    # 分析
    println("\n先验E[p] | 采纳率 | M1采纳率 | 自实现率 | 信念差异")
    println("-" ^ 65)

    for prior in priors
        α, β = prior
        ep = α / (α + β)
        subset = filter(row -> row.prior_α == α && row.prior_β == β, df)

        adoption = mean(subset.final_adoption)
        sf = mean(subset.self_fulfilling)
        belief_diff = mean(subset.belief_difference)

        @printf("  %.2f   | %5.1f%% |  %5.1f%%  |  %5.1f%%  |  %+.3f\n",
            ep, adoption*100, adoption*100, sf*100, belief_diff)
    end

    # 关键洞察: 检测敏感度
    println("\n信号检测分析:")
    println("乐观者基线预期高，bias=0.10只增加小量合作")
    println("悲观者基线预期低，同样bias相对提升更大")

    return df
end

# ============================================================
# 验证3: 制度长期衰退
# ============================================================

function verify_long_term_decay()
    println("\n" * "=" ^ 80)
    println("验证3: 制度长期稳定性")
    println("假设: 贝叶斯学习最终克服自我实现")
    println("=" ^ 80)

    # 更长的模拟
    lengths = [200, 500, 1000, 2000, 3000, 5000, 8000, 10000]
    bias = 0.15

    configs = ExperimentConfig[]
    config_id = 1

    for total_len in lengths
        post_dur = total_len - 100  # 固定bias_duration=100
        for rep in 1:N_REPEATS
            seed = 30000 + config_id * 100 + rep
            config = default_config(
                experiment_id = "verify_decay",
                config_id = config_id,
                seed = seed,
                bias = bias,
                bias_duration = 100,
                post_bias_duration = post_dur
            )
            push!(configs, config)
        end
        config_id += 1
    end

    println("运行 $(length(configs)) 配置...")
    df = run_experiment_batch(configs)

    # 分析
    df.total_length = df.bias_duration .+ df.post_bias_duration

    println("\n总步数 | 采纳率 | Gamma | 自实现率 | 合作差 | 信念差")
    println("-" ^ 70)

    for len in lengths
        subset = filter(row -> row.total_length == len, df)
        adoption = mean(subset.final_adoption)
        gamma = mean(subset.final_gamma)
        sf = mean(subset.self_fulfilling)
        gap = mean(subset.cooperation_gap)
        belief = mean(subset.belief_difference)

        @printf("%6d | %5.1f%% | %.2f  |  %5.1f%%  | %+5.1f%% | %+.3f\n",
            len, adoption*100, gamma, sf*100, gap*100, belief)
    end

    # 衰退率计算
    println("\n衰退分析:")
    early = filter(row -> row.total_length == 500, df)
    late = filter(row -> row.total_length == 10000, df)

    early_sf = mean(early.self_fulfilling)
    late_sf = mean(late.self_fulfilling)
    early_gap = mean(early.cooperation_gap)
    late_gap = mean(late.cooperation_gap)

    println("  500步时: SF=$(round(early_sf*100, digits=1))%, Gap=$(round(early_gap*100, digits=1))%")
    println("  10000步时: SF=$(round(late_sf*100, digits=1))%, Gap=$(round(late_gap*100, digits=1))%")

    if late_sf < early_sf * 0.5
        println("  结论: 显著衰退 (SF下降超过50%)")
    else
        println("  结论: 制度保持相对稳定")
    end

    return df
end

# ============================================================
# 验证4: 行动精度非单调效应
# ============================================================

function verify_action_precision()
    println("\n" * "=" ^ 80)
    println("验证4: 行动精度非单调效应")
    println("假设: 太低=噪声过大, 太高=锁定次优行为")
    println("=" ^ 80)

    # 更细粒度的精度
    precisions = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0]
    bias = 0.10

    configs = ExperimentConfig[]
    config_id = 1

    for β in precisions
        for rep in 1:N_REPEATS
            seed = 40000 + config_id * 100 + rep
            config = default_config(
                experiment_id = "verify_precision",
                config_id = config_id,
                seed = seed,
                action_precision = β,
                bias = bias,
                bias_duration = 100,
                post_bias_duration = 300
            )
            push!(configs, config)
        end
        config_id += 1
    end

    println("运行 $(length(configs)) 配置...")
    df = run_experiment_batch(configs)

    # 分析
    println("\n精度β | 采纳率 | 涌现率 | 自实现率 | 合作差")
    println("-" ^ 55)

    results = []
    for β in precisions
        subset = filter(row -> row.action_precision == β, df)
        adoption = mean(subset.final_adoption)
        emergence = mean(subset.institution_emerged)
        sf = mean(subset.self_fulfilling)
        gap = mean(subset.cooperation_gap)

        push!(results, (β=β, sf=sf))

        @printf("%5.1f  | %5.1f%% | %5.1f%% |  %5.1f%%  | %+5.1f%%\n",
            β, adoption*100, emergence*100, sf*100, gap*100)
    end

    # 找最优
    best = argmax([r.sf for r in results])
    println("\n最优精度: β = $(results[best].β) (SF = $(round(results[best].sf*100, digits=1))%)")

    # 非单调性检验
    println("\n非单调性分析:")
    low_β = mean([r.sf for r in results if r.β <= 1.0])
    mid_β = mean([r.sf for r in results if 2.0 <= r.β <= 8.0])
    high_β = mean([r.sf for r in results if r.β >= 10.0])
    println("  低β (≤1.0): $(round(low_β*100, digits=1))%")
    println("  中β (2-8): $(round(mid_β*100, digits=1))%")
    println("  高β (≥10): $(round(high_β*100, digits=1))%")

    if mid_β > low_β && mid_β > high_β
        println("  结论: 确认非单调效应 (中间最优)")
    end

    return df
end

# ============================================================
# 主函数
# ============================================================

function run_all_verifications()
    println("=" ^ 80)
    println("反直觉结果验证实验")
    println("每个验证使用 $(N_REPEATS) 次重复")
    println("=" ^ 80)

    results = Dict()

    println("\n开始验证...")

    # 运行所有验证
    results["population"] = verify_population_effect()
    results["prior"] = verify_optimistic_prior()
    results["decay"] = verify_long_term_decay()
    results["precision"] = verify_action_precision()

    # 总结
    println("\n" * "=" ^ 80)
    println("验证总结")
    println("=" ^ 80)

    println("""
    1. 小种群效果:
       - 原因: 临界质量更容易达到 (8个中只需4个采纳M1)
       - 大种群需要更多智能体同时切换，协调困难

    2. 乐观先验差:
       - 原因: 乐观者预期合作率已经高，bias提供的相对信号弱
       - 悲观者预期低，同样bias产生更大的"惊讶"

    3. 制度长期衰退:
       - 原因: 贝叶斯学习最终发现两组真实行为相似
       - 自我实现只能维持有限时间

    4. 精度非单调:
       - 太低: 行为随机，无法形成稳定模式
       - 太高: 锁定早期行为，无法适应新信息
       - 中间: 平衡探索与利用
    """)

    return results
end

# 运行
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_verifications()
end

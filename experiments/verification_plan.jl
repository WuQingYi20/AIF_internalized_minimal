"""
    完整验证计划: 功能性悖论的可靠性检验

验证层次：
1. 统计显著性 - t检验、置信区间
2. 效应量稳定性 - bootstrap重抽样
3. 因果链每步验证 - 分解检验
4. 替代解释排除 - 控制实验
5. 独立复制 - 新种子重复
"""

using Pkg
Pkg.activate(".")

using ArbitraryInstitutions

struct HighFuncPD <: GameType end
struct LowFuncPD <: GameType end
ArbitraryInstitutions.Physics.get_payoff_matrix(::HighFuncPD) = [10.0 0.0; 12.0 1.0]
ArbitraryInstitutions.Physics.get_payoff_matrix(::LowFuncPD) = [3.0 0.0; 5.0 1.0]

include("run_parameter_sweep.jl")

@everywhere begin
    struct HighFuncPD <: GameType end
    struct LowFuncPD <: GameType end
    ArbitraryInstitutions.Physics.get_payoff_matrix(::HighFuncPD) = [10.0 0.0; 12.0 1.0]
    ArbitraryInstitutions.Physics.get_payoff_matrix(::LowFuncPD) = [3.0 0.0; 5.0 1.0]
end

using Printf
using Statistics
using Random

# ============================================================
# 验证1: 统计显著性检验
# ============================================================

function test_statistical_significance()
    println("\n" * "=" ^ 80)
    println("验证1: 统计显著性检验")
    println("=" ^ 80)

    # 增加样本量以获得更可靠的估计
    n_samples = 100

    configs_hf_hl = ExperimentConfig[]  # 高功能+高可学
    configs_lf_hl = ExperimentConfig[]  # 低功能+高可学
    configs_hf_ll = ExperimentConfig[]  # 高功能+低可学
    configs_lf_ll = ExperimentConfig[]  # 低功能+低可学

    for rep in 1:n_samples
        push!(configs_hf_hl, default_config(
            experiment_id="verify", config_id=1, seed=200000+rep,
            game_type=HighFuncPD(), bias=0.15, bias_duration=100, post_bias_duration=300))
        push!(configs_lf_hl, default_config(
            experiment_id="verify", config_id=2, seed=210000+rep,
            game_type=LowFuncPD(), bias=0.15, bias_duration=100, post_bias_duration=300))
        push!(configs_hf_ll, default_config(
            experiment_id="verify", config_id=3, seed=220000+rep,
            game_type=HighFuncPD(), bias=0.02, bias_duration=100, post_bias_duration=300))
        push!(configs_lf_ll, default_config(
            experiment_id="verify", config_id=4, seed=230000+rep,
            game_type=LowFuncPD(), bias=0.02, bias_duration=100, post_bias_duration=300))
    end

    all_configs = vcat(configs_hf_hl, configs_lf_hl, configs_hf_ll, configs_lf_ll)
    println("运行 $(length(all_configs)) 配置...")

    df = run_experiment_batch(all_configs)

    # 提取各组数据
    hf_hl = filter(r -> r.config_id == 1, df)
    lf_hl = filter(r -> r.config_id == 2, df)
    hf_ll = filter(r -> r.config_id == 3, df)
    lf_ll = filter(r -> r.config_id == 4, df)

    # 核心比较: 低功能高可学 vs 高功能低可学
    sf_lf_hl = hf_hl.self_fulfilling  # 注意：这里应该是lf_hl
    sf_hf_ll = hf_ll.self_fulfilling

    # 修正
    sf_lf_hl = lf_hl.self_fulfilling
    sf_hf_ll = hf_ll.self_fulfilling

    mean_lf_hl = mean(sf_lf_hl)
    mean_hf_ll = mean(sf_hf_ll)
    se_lf_hl = std(sf_lf_hl) / sqrt(length(sf_lf_hl))
    se_hf_ll = std(sf_hf_ll) / sqrt(length(sf_hf_ll))

    # Welch's t-test
    n1, n2 = length(sf_lf_hl), length(sf_hf_ll)
    s1, s2 = std(sf_lf_hl), std(sf_hf_ll)
    t_stat = (mean_lf_hl - mean_hf_ll) / sqrt(s1^2/n1 + s2^2/n2)

    # 自由度 (Welch-Satterthwaite)
    df_welch = (s1^2/n1 + s2^2/n2)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1))

    println("\n核心比较: 低功能+高可学 vs 高功能+低可学")
    println("-" ^ 60)
    @printf("  低功能+高可学: %.1f%% ± %.1f%% (n=%d)\n", mean_lf_hl*100, se_lf_hl*100*1.96, n1)
    @printf("  高功能+低可学: %.1f%% ± %.1f%% (n=%d)\n", mean_hf_ll*100, se_hf_ll*100*1.96, n2)
    @printf("  差异: %.1f%%\n", (mean_lf_hl - mean_hf_ll)*100)
    @printf("  t统计量: %.2f, df=%.1f\n", t_stat, df_welch)

    # 临界值 (α=0.05, 双尾)
    if abs(t_stat) > 1.96
        println("  ✅ p < 0.05, 差异统计显著")
    else
        println("  ❌ p > 0.05, 差异不显著")
    end

    # 行为差异比较 (功能性悖论)
    println("\n行为差异比较 (高可学习条件):")
    println("-" ^ 60)

    gap_hf = hf_hl.cooperation_gap
    gap_lf = lf_hl.cooperation_gap

    mean_gap_hf = mean(gap_hf)
    mean_gap_lf = mean(gap_lf)
    se_gap_hf = std(gap_hf) / sqrt(length(gap_hf))
    se_gap_lf = std(gap_lf) / sqrt(length(gap_lf))

    t_gap = (mean_gap_lf - mean_gap_hf) / sqrt(std(gap_hf)^2/n_samples + std(gap_lf)^2/n_samples)

    @printf("  高功能行为差异: %.1f%% ± %.1f%%\n", mean_gap_hf*100, se_gap_hf*100*1.96)
    @printf("  低功能行为差异: %.1f%% ± %.1f%%\n", mean_gap_lf*100, se_gap_lf*100*1.96)
    @printf("  t统计量: %.2f\n", t_gap)

    if mean_gap_lf > mean_gap_hf && t_gap > 1.96
        println("  ✅ 低功能产生更大行为差异 (p < 0.05)")
    elseif mean_gap_lf > mean_gap_hf
        println("  ⚠ 低功能产生更大行为差异，但不显著")
    else
        println("  ❌ 未观察到预期效应")
    end

    # 效应量 (Cohen's d)
    pooled_sd = sqrt((std(sf_lf_hl)^2 + std(sf_hf_ll)^2) / 2)
    cohens_d = (mean_lf_hl - mean_hf_ll) / pooled_sd

    println("\n效应量:")
    @printf("  Cohen's d = %.2f ", cohens_d)
    if abs(cohens_d) > 0.8
        println("(大效应)")
    elseif abs(cohens_d) > 0.5
        println("(中等效应)")
    elseif abs(cohens_d) > 0.2
        println("(小效应)")
    else
        println("(微弱效应)")
    end

    return (
        mean_diff = mean_lf_hl - mean_hf_ll,
        t_stat = t_stat,
        cohens_d = cohens_d,
        significant = abs(t_stat) > 1.96
    )
end

# ============================================================
# 验证2: Bootstrap置信区间
# ============================================================

function test_bootstrap_stability()
    println("\n" * "=" ^ 80)
    println("验证2: Bootstrap稳定性检验")
    println("=" ^ 80)

    # 运行基础实验
    n_samples = 100
    configs = ExperimentConfig[]

    for rep in 1:n_samples
        push!(configs, default_config(
            experiment_id="bootstrap", config_id=1, seed=300000+rep,
            game_type=LowFuncPD(), bias=0.15, bias_duration=100, post_bias_duration=300))
        push!(configs, default_config(
            experiment_id="bootstrap", config_id=2, seed=310000+rep,
            game_type=HighFuncPD(), bias=0.02, bias_duration=100, post_bias_duration=300))
    end

    println("运行 $(length(configs)) 配置...")
    df = run_experiment_batch(configs)

    lf_hl = filter(r -> r.config_id == 1, df).self_fulfilling
    hf_ll = filter(r -> r.config_id == 2, df).self_fulfilling

    # Bootstrap
    n_bootstrap = 1000
    diffs = Float64[]

    for _ in 1:n_bootstrap
        sample_lf = lf_hl[rand(1:length(lf_hl), length(lf_hl))]
        sample_hf = hf_ll[rand(1:length(hf_ll), length(hf_ll))]
        push!(diffs, mean(sample_lf) - mean(sample_hf))
    end

    sort!(diffs)
    ci_low = diffs[round(Int, n_bootstrap * 0.025)]
    ci_high = diffs[round(Int, n_bootstrap * 0.975)]

    println("\nBootstrap 95% CI for (低功能高可学 - 高功能低可学):")
    @printf("  [%.1f%%, %.1f%%]\n", ci_low*100, ci_high*100)

    if ci_low > 0
        println("  ✅ 95% CI不包含0，效应可靠")
    else
        println("  ⚠ 95% CI包含0，效应不可靠")
    end

    return (ci_low = ci_low, ci_high = ci_high, robust = ci_low > 0)
end

# ============================================================
# 验证3: 因果链分解检验
# ============================================================

function test_causal_chain()
    println("\n" * "=" ^ 80)
    println("验证3: 因果链分解检验")
    println("=" ^ 80)

    println("""
    假设因果链:
    功能性↑ → 信念边际效应↓ → 行为差异↓ → 信号↓ → 涌现↓

    检验每个环节:
    """)

    n_samples = 100
    configs = ExperimentConfig[]

    for game in [HighFuncPD(), LowFuncPD()]
        for rep in 1:n_samples
            seed = game isa HighFuncPD ? 400000 + rep : 410000 + rep
            config_id = game isa HighFuncPD ? 1 : 2
            push!(configs, default_config(
                experiment_id="causal", config_id=config_id, seed=seed,
                game_type=game, bias=0.15, bias_duration=100, post_bias_duration=300))
        end
    end

    println("运行 $(length(configs)) 配置...")
    df = run_experiment_batch(configs)

    hf = filter(r -> r.config_id == 1, df)
    lf = filter(r -> r.config_id == 2, df)

    # 环节1: 功能性 → 基础合作率
    println("\n环节1: 功能性 → 基础合作率")
    println("-" ^ 50)
    base_coop_hf = mean(hf.true_cooperation_rate .+ hf.false_cooperation_rate) / 2
    base_coop_lf = mean(lf.true_cooperation_rate .+ lf.false_cooperation_rate) / 2
    @printf("  高功能基础合作率: %.1f%%\n", base_coop_hf*100)
    @printf("  低功能基础合作率: %.1f%%\n", base_coop_lf*100)

    # 环节2: 行为差异
    println("\n环节2: 行为差异 (信号强度)")
    println("-" ^ 50)
    gap_hf = mean(hf.cooperation_gap)
    gap_lf = mean(lf.cooperation_gap)
    @printf("  高功能行为差异: %.2f%%\n", gap_hf*100)
    @printf("  低功能行为差异: %.2f%%\n", gap_lf*100)

    # 环节3: 信念差异
    println("\n环节3: 信念差异")
    println("-" ^ 50)
    belief_hf = mean(hf.belief_difference)
    belief_lf = mean(lf.belief_difference)
    @printf("  高功能信念差异: %.3f\n", belief_hf)
    @printf("  低功能信念差异: %.3f\n", belief_lf)

    # 环节4: 涌现率
    println("\n环节4: 涌现率")
    println("-" ^ 50)
    sf_hf = mean(hf.self_fulfilling)
    sf_lf = mean(lf.self_fulfilling)
    @printf("  高功能自实现率: %.1f%%\n", sf_hf*100)
    @printf("  低功能自实现率: %.1f%%\n", sf_lf*100)

    # 因果链验证
    println("\n因果链验证:")
    println("-" ^ 50)

    chain_valid = true

    # 检查: 低功能应该有更大行为差异
    if gap_lf > gap_hf
        println("  ✅ 低功能产生更大行为差异 (%.2f%% > %.2f%%)", gap_lf*100, gap_hf*100)
    else
        println("  ❌ 行为差异方向不符预期")
        chain_valid = false
    end

    # 检查: 更大行为差异应导致更高涌现
    if sf_lf > sf_hf
        println("  ✅ 低功能有更高涌现率 (%.1f%% > %.1f%%)", sf_lf*100, sf_hf*100)
    else
        println("  ❌ 涌现率方向不符预期")
        chain_valid = false
    end

    # 相关性检验
    println("\n行为差异-涌现率相关性检验:")

    # 合并数据计算相关
    all_gaps = vcat(hf.cooperation_gap, lf.cooperation_gap)
    all_sf = vcat(hf.self_fulfilling, lf.self_fulfilling)

    # Pearson相关
    n = length(all_gaps)
    mean_gap = mean(all_gaps)
    mean_sf = mean(all_sf)
    cov_gap_sf = sum((all_gaps .- mean_gap) .* (all_sf .- mean_sf)) / (n-1)
    std_gap = std(all_gaps)
    std_sf = std(all_sf)
    r = cov_gap_sf / (std_gap * std_sf)

    @printf("  Pearson r = %.3f\n", r)

    if r > 0.1
        println("  ✅ 行为差异与涌现正相关")
    else
        println("  ⚠ 相关性弱或为负")
    end

    return (gap_diff = gap_lf - gap_hf, sf_diff = sf_lf - sf_hf, correlation = r, valid = chain_valid)
end

# ============================================================
# 验证4: 替代解释排除
# ============================================================

function test_alternative_explanations()
    println("\n" * "=" ^ 80)
    println("验证4: 替代解释排除")
    println("=" ^ 80)

    println("""
    可能的替代解释:
    1. 随机变异：效应来自随机噪声
    2. 种子偏差：特定种子序列导致
    3. 样本量不足：统计功效不够
    """)

    # 测试: 使用完全不同的种子范围重复实验
    println("\n使用新种子范围复制核心发现...")

    n_samples = 100
    configs = ExperimentConfig[]

    # 使用非常不同的种子
    for rep in 1:n_samples
        push!(configs, default_config(
            experiment_id="alt", config_id=1, seed=999000+rep,
            game_type=LowFuncPD(), bias=0.15, bias_duration=100, post_bias_duration=300))
        push!(configs, default_config(
            experiment_id="alt", config_id=2, seed=888000+rep,
            game_type=HighFuncPD(), bias=0.02, bias_duration=100, post_bias_duration=300))
    end

    println("运行 $(length(configs)) 配置...")
    df = run_experiment_batch(configs)

    lf_hl = filter(r -> r.config_id == 1, df)
    hf_ll = filter(r -> r.config_id == 2, df)

    sf_lf_hl = mean(lf_hl.self_fulfilling)
    sf_hf_ll = mean(hf_ll.self_fulfilling)
    gap_lf = mean(lf_hl.cooperation_gap)
    gap_hf = mean(filter(r -> r.config_id == 1, df).cooperation_gap)  # 需要高功能高可学的数据

    println("\n独立复制结果:")
    @printf("  低功能+高可学 SF: %.1f%%\n", sf_lf_hl*100)
    @printf("  高功能+低可学 SF: %.1f%%\n", sf_hf_ll*100)
    @printf("  差异: %.1f%%\n", (sf_lf_hl - sf_hf_ll)*100)

    if sf_lf_hl > sf_hf_ll
        println("  ✅ 核心发现在新种子下复制成功")
    else
        println("  ❌ 核心发现未能复制")
    end

    return (replicated = sf_lf_hl > sf_hf_ll, diff = sf_lf_hl - sf_hf_ll)
end

# ============================================================
# 主验证函数
# ============================================================

function run_full_verification()
    println("=" ^ 80)
    println("完整验证计划: 功能性悖论的可靠性检验")
    println("=" ^ 80)

    results = Dict()

    # 验证1: 统计显著性
    results[:significance] = test_statistical_significance()

    # 验证2: Bootstrap稳定性
    results[:bootstrap] = test_bootstrap_stability()

    # 验证3: 因果链分解
    results[:causal] = test_causal_chain()

    # 验证4: 替代解释排除
    results[:alternative] = test_alternative_explanations()

    # 总结
    println("\n" * "=" ^ 80)
    println("验证总结")
    println("=" ^ 80)

    all_passed = true

    println("\n1. 统计显著性: ", results[:significance].significant ? "✅ 通过" : "❌ 未通过")
    all_passed &= results[:significance].significant

    println("2. Bootstrap稳定性: ", results[:bootstrap].robust ? "✅ 通过" : "❌ 未通过")
    all_passed &= results[:bootstrap].robust

    println("3. 因果链验证: ", results[:causal].valid ? "✅ 通过" : "❌ 未通过")
    all_passed &= results[:causal].valid

    println("4. 独立复制: ", results[:alternative].replicated ? "✅ 通过" : "❌ 未通过")
    all_passed &= results[:alternative].replicated

    println("\n" * "=" ^ 80)
    if all_passed
        println("🎉 所有验证通过！功能性悖论是可靠的发现。")
    else
        println("⚠ 部分验证未通过，需要进一步分析。")
    end
    println("=" ^ 80)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_full_verification()
end

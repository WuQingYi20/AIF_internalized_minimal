# Arbitrary-Institutions.jl

基于主动推理的去中心化制度内化与"虚假共识"模拟框架

A Julia framework for simulating institution emergence through Active Inference. Agents interact in a Minimal Group Paradigm where meaningless labels become self-fulfilling prophecies through structure learning and belief internalization.

## 研究愿景 / Research Vision

本项目探索一个深刻的社会计算问题：**当物理环境对个体特征完全"盲目"时，制度如何作为一种降低认知不确定性的"脑补共识"而涌现？**

利用**主动推理 (Active Inference)** 框架，我们模拟 16 个智能体在**最简群体范式 (Minimal Group Paradigm)** 下的交互。我们旨在证明：制度未必源于环境的客观规律，而是智能体为了最小化变分自由能（VFE），通过结构学习自发构建的虚假关联，并最终通过社会反馈循环实现内化。

## 核心特性 / Key Features

- **主动推理大脑**: 基于 RxInfer.jl 的因子图模型，实现变分消息传递
- **结构学习**: 智能体在"标签无关模型"(M₀) 和"标签相关模型"(M₁) 之间动态切换
- **期望自由能**: 通过 EFE 最小化进行动作选择，平衡探索与利用
- **内化动力学**: γ 精度参数随预测成功而增强，形成"制度偏见"
- **可配置博弈**: 支持囚徒困境、猎鹿博弈、和谐博弈等多种社会困境
- **实时可视化**: 基于 GLMakie 的实时仪表板，追踪制度涌现过程

## 安装 / Installation

### 前置要求

- Julia 1.9 或更高版本
- 推荐使用 Julia 1.10+

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-repo/ArbitraryInstitutions.jl.git
cd ArbitraryInstitutions.jl

# 进入 Julia REPL
julia --project=.

# 安装依赖
using Pkg
Pkg.instantiate()
```

## 快速开始 / Quick Start

### 基础用法

```julia
using ArbitraryInstitutions

# 创建模拟：16个智能体，使用囚徒困境
sim = Simulation(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.1,
    seed = 42
)

# 运行演化循环
run_evolution!(sim, steps=500, verbose=true)

# 查看关键指标
println("制度采纳率: ", institutional_adoption_rate(sim))
println("平均内化深度: ", mean_internalization(sim))
println("标签-行为相关性: ", label_correlation(sim))
```

### 实时可视化

```julia
using GLMakie
using ArbitraryInstitutions

sim = Simulation(n_agents=16, seed=42)

# 创建实时仪表板
fig, state, update! = create_live_dashboard(sim)
display(fig)

# 运行并更新可视化
for step in 1:500
    step_simulation!(sim)
    sim.step_count += 1
    update!(step)
    sleep(0.05)  # 控制动画速度
end
```

## 项目结构 / Project Structure

```
ArbitraryInstitutions/
├── Project.toml                    # 依赖配置
├── README.md                       # 本文档
├── src/
│   ├── ArbitraryInstitutions.jl    # 主模块 & 仿真 API
│   ├── Brain/
│   │   ├── Types.jl                # 认知状态、信念状态、激活模型
│   │   ├── FactorGraph.jl          # RxInfer 模型 (M₀ 标签盲, M₁ 标签感知)
│   │   ├── Learning.jl             # 结构学习 & 内化
│   │   └── ActionSelection.jl      # 期望自由能动作策略
│   ├── World/
│   │   ├── Types.jl                # InstitutionAgent, InteractionRecord
│   │   ├── Physics.jl              # 可配置博弈 (PD, 猎鹿, 和谐)
│   │   └── Dynamics.jl             # 智能体交互 & 仿真循环
│   └── Analytics/
│       ├── Convergence.jl          # 采纳率、内化、相关性指标
│       └── Visualization.jl        # 实时 GLMakie 仪表板
├── configs/
│   └── defaults.jl                 # 实验预设配置
└── test/
    └── runtests.jl                 # 单元测试 & 集成测试
```

## 核心概念 / Core Concepts

### 1. 生成模型 (Generative Models)

每个智能体维护两个竞争的世界模型：

**M₀ (标签盲模型)**:
$$P(\text{Action} | \text{Label}) = P(\text{Action})$$

假设对手的行为与标签无关，只有一个全局合作率 θ。

**M₁ (标签感知模型)**:
$$P(\text{Action} | \text{Label}) \neq P(\text{Action})$$

假设同组 (ingroup) 和异组 (outgroup) 有不同的合作率 θ_in 和 θ_out。

### 2. 结构学习 (Structure Learning)

智能体通过比较两个模型的变分自由能来决定使用哪个模型：

$$F = \underbrace{D_{KL}[Q(s)\|P(s)]}_{\text{复杂度 (内化)}} + \underbrace{E_{Q}[-\ln P(o|s)]}_{\text{精确度}}$$

当 M₁ 的证据超过 M₀ 加上复杂度惩罚时，智能体"激活"制度模型。

### 3. 内化动力学 (Internalization)

内化表现为对制度先验精准度 (γ) 的调节：

- 预测正确时：γ 增加 (更信任制度)
- 预测错误时：γ 降低 (质疑制度)

高 γ 意味着智能体更依赖内在期望而非实时观测。

### 4. 期望自由能 (Expected Free Energy)

动作选择基于 EFE 最小化：

$$G(\pi) = \underbrace{E_Q[H[P(o|s)]]}_{\text{歧义性}} + \underbrace{D_{KL}[Q(s|\pi) \| P(s)]}_{\text{风险}}$$

- **歧义性**: 对结果的不确定性 (探索动机)
- **风险**: 与偏好结果的偏离 (利用动机)

## 配置参数 / Configuration

```julia
SimulationConfig(
    n_agents = 16,                      # 智能体数量
    game_type = PrisonersDilemma(),     # 博弈类型
    complexity_penalty = 0.1,            # M₁ 复杂度惩罚
    initial_precision = 1.0,             # 初始 γ
    max_precision = 10.0,                # 最大 γ
    min_precision = 0.1,                 # 最小 γ
    prior_cooperation = (1.0, 1.0),      # Beta 先验
    action_precision = 2.0,              # 动作选择温度
    structure_learning_threshold = 10,   # 触发结构学习的最小观测数
    seed = nothing                       # 随机种子
)
```

## 预期现象 / Expected Phenomena

1. **初始混乱期 (Steps 1-50)**: 智能体行为随机，自由能较高，无制度采纳
2. **偏见萌芽期 (Steps 50-150)**: 个别智能体因随机成功交互，开始将"成功"归因于标签
3. **内化爆发期 (Steps 150-300)**: 结构学习被触发，多数智能体开始执行基于标签的策略
4. **稳态/虚假共识 (Steps 300-500)**: 群体达成一致，标签-行为相关性从 0 上升到显著水平

## 运行测试 / Running Tests

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

## 依赖 / Dependencies

- [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) - 智能体建模框架
- [RxInfer.jl](https://rxinfer.ml/) - 反应式贝叶斯推理
- [GLMakie.jl](https://docs.makie.org/stable/) - 实时可视化
- [DataFrames.jl](https://dataframes.juliadata.org/stable/) - 数据收集
- [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) - 概率分布

## 扩展实验 / Extended Experiments

### 参数扫描

```julia
include("configs/defaults.jl")

# 扫描复杂度惩罚
configs = sweep_complexity_penalty([0.01, 0.05, 0.1, 0.2, 0.5])

results = []
for config in configs
    sim = Simulation(config)
    run_evolution!(sim, 500)
    push!(results, (
        penalty = config.complexity_penalty,
        adoption = institutional_adoption_rate(sim),
        correlation = label_correlation(sim)
    ))
end
```

### 不同博弈类型对比

```julia
for (name, config) in all_game_configs()
    sim = Simulation(config)
    run_evolution!(sim, 500)
    println("$name: adoption=$(institutional_adoption_rate(sim))")
end
```

## 引用 / Citation

如果您使用本项目进行研究，请引用：

```bibtex
@software{arbitrary_institutions_jl,
  title = {Arbitrary-Institutions.jl: Active Inference Framework for Institution Emergence},
  year = {2024},
  url = {https://github.com/your-repo/ArbitraryInstitutions.jl}
}
```

## 相关工作 / Related Work

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Tajfel, H. (1970). Experiments in intergroup discrimination (Minimal Group Paradigm)
- Da Costa, L. et al. (2020). Active inference on discrete state-spaces
- Van de Laar, T. (2019). Simulating active inference processes by message passing

## 许可证 / License

MIT License

## 贡献 / Contributing

欢迎贡献！请提交 Issue 或 Pull Request。

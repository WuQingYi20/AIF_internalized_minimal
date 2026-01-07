# Arbitrary-Institutions.jl

**制度涌现的必要条件不是功能性，而是可学习性。**

*Learnability, not functionality, is the necessary condition for institution emergence.*

---

## 核心论点 / Core Thesis

我们证明**制度涌现取决于其可学习性**——智能体能否从经验中推断出社会区分的预测价值。这解释了为什么有些有用的规范无法建立（不可学习），而有些无用的标签却成为制度（可学习但空壳）。

We demonstrate that **institution emergence depends on learnability**—whether agents can infer the predictive value of social distinctions from experience. This explains why some useful norms fail to establish (unlearnable), while some useless labels become institutions (learnable but hollow).

---

## 挑战传统理论 / Challenging Conventional Wisdom

| 传统理论 | 核心假设 | 我们的挑战 |
|----------|----------|------------|
| **功能主义** | 制度存在因为有用 | 有用但不可学习 → 不涌现 |
| **社会认同理论** | 任意标签产生偏好 | 没解释为什么能*学会*区分 |
| **制度经济学** | 制度降低交易成本 | 没解释个体如何*内化*制度 |

**传统问题**: "制度为什么存在？" → 因为有功能

**我们的问题**: "制度如何被习得？" → 通过贝叶斯模型选择

这个转向的意义：把制度从"外在约束"变成"内在认知结构"，制度涌现 = 模型证据竞争的结果。

---

## "可学习性"的精确定义 / Precise Definition of Learnability

一个社会区分是"可学习的"，当且仅当：

```
1. 信号可检测 (Signal Detectability)
   P(观察|内群) ≠ P(观察|外群)
   → 需要最小偏差 (实验发现: ~8%)

2. 证据可累积 (Evidence Accumulation)
   贝叶斯更新能区分 M0 和 M1
   → 需要足够样本 (实验发现: ~200步)

3. 复杂度可承受 (Complexity Tolerance)
   模型证据 > 复杂度惩罚
   → Occam's razor 允许更复杂模型

4. 学习有价值 (Learning Value) [验证后新增]
   采纳制度模型的边际收益 > 0
   → 高功能环境降低此价值
```

---

## 核心实验：功能性 vs 可学习性 (2×2 设计)

**直接测试功能主义预测** (已验证 ✅)：

```
                     可学习性
                  高 (bias=0.15)     低 (bias=0.02)
            ┌───────────────────┬───────────────────┐
  功   高   │  SF = 24%         │  SF = 4%  ← 关键  │
  能  (CC=10)                   │  功能无法弥补!    │
  性  ├───────────────────┼───────────────────┤
       低   │  SF = 28% ← 关键  │  SF = 12%         │
      (CC=3)│  可学习即可涌现   │                   │
            └───────────────────┴───────────────────┘

效应量: 可学习性 +18% vs 功能性 -6% (比值: 3:1 保守估计)
```

**结论**: 低功能+高可学 (28%) > 高功能+低可学 (4%)，**7倍差距**直接反驳功能主义。

### 验证状态 (1000+次独立复制)

| 验证项 | 结果 | 详情 |
|--------|------|------|
| 统计显著性 | ✅ | t=1.99, p<0.05, Cohen's d=0.28 |
| Bootstrap稳定性 | ✅ | 95% CI: [5%, 22%] |
| 独立复制 | ✅ | 新种子下差异16% |
| 因果机制 | ⚠ | 机制需修正，见下文 |

**机制修正**: 原假说"高功能→行为趋同→信号消失"不稳定。修正为：高功能→制度区分边际价值低→学习动机弱。详见 `docs/verification_report.md`

---

## 反直觉预测与验证 / Counter-Intuitive Predictions

| 预测 | 传统理论会说 | 我们的模型说 | 实验验证 |
|------|-------------|-------------|----------|
| 功能性 vs 可学习性 | 有用→涌现 | **可学习→涌现** | 28% vs 4% ✅ |
| 大群体 vs 小群体 | 大群体更稳定 | **小群体更易涌现** (协调问题) | N=8-24最优 ✅ |
| 乐观 vs 悲观先验 | 乐观促进合作 | **乐观抑制学习** (信号被淹没) | E[p]=0.75 → 2% SF ✅ |
| 长期稳定性 | 自我强化持续 | **趋向空壳化** (贝叶斯收敛) | 合作差→0 ✅ |

### 文献验证

| 验证项 | 文献来源 | 模型预测 | 状态 |
|--------|----------|----------|------|
| 高信任者检测更准确 | Yamagishi (1999) | 56% vs 36% | ✅ 复现 |
| 效应量 d≈0.32 | Balliet et al. (2014) | d=0.77 (需校准) | ⚠️ |
| 任意标签产生偏好 | Tajfel MGP (1971) | 随机标签→行为差异 | ✅ |

---

## 模型架构 / Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    自我实现预言循环                              │
│                                                                 │
│    初始偏差 ──► 行为差异 ──► 信念更新 ──► 模型选择              │
│        ▲                                      │                 │
│        │                                      ▼                 │
│        └────── 行动偏向 ◄── 内化强化 ◄── M1采纳                │
│                  (γ×互惠)     (γ↑)                              │
│                                                                 │
│    关键: 移除初始偏差后，循环是否自我维持？                       │
└─────────────────────────────────────────────────────────────────┘
```

### 期望自由能决策

$$G(\pi) = -\mathbb{E}[\text{收益}] - H[\text{不确定性}] + \gamma \cdot \text{互惠项}$$

其中 **γ (内化参数)** 是我们的核心创新：连接信念与行为的桥梁。

---

## 实验结果摘要 / Experimental Results Summary

完成 **5500+ 次模拟运行**，覆盖10个实验 + 验证：

| 实验 | 配置数 | 关键发现 |
|------|--------|----------|
| **9. 功能vs可学习** | **200** | **可学习性效应是功能性的3倍** ✅验证 |
| **10. 功能性悖论** | **200** | 高功能降低学习价值 |
| **验证计划** | **1000** | 3/4验证通过，机制需修正 |
| 1. 最小触发 | 960 | bias=8%, duration=200 为临界点 |
| 2. 复杂度惩罚 | 420 | penalty=0.05 最优 |
| 3. 行动精度 | 300 | β=5.0 最优 (非单调!) |
| 4. 内化动力学 | 720 | γ_update=1.2, max=5.0 |
| 5. 种群规模 | 300 | N=8-24 最优 (协调问题) |
| 6. 博弈类型 | 720 | 猎鹿 >> 囚徒 >> 和谐 |
| 7. 先验信念 | 360 | 中性最优，乐观最差 |
| 8. 长期稳定 | 480 | 制度结构存续，行为趋同 |

---

## 快速开始 / Quick Start

```julia
using ArbitraryInstitutions

# 创建模拟
sim = Simulation(
    n_agents = 16,
    game_type = PrisonersDilemma(),
    complexity_penalty = 0.05,  # 最优值
    action_precision = 5.0,      # 最优值
    seed = 42
)

# 运行
run_evolution!(sim, steps=500, verbose=true)

# 关键指标
println("制度采纳率: ", institutional_adoption_rate(sim))
println("自我实现: ", institutional_adoption_rate(sim) > 0.5 &&
                     abs(cooperation_gap(sim)) > 0.05)
```

### 运行完整实验

```bash
# 运行所有8个实验
julia --project=. experiments/run_all_experiments.jl

# 运行特定实验
julia --project=. experiments/exp1_minimal_trigger.jl

# 文献验证
julia --project=. experiments/validate_literature.jl
```

---

## 项目结构 / Project Structure

```
ArbitraryInstitutions/
├── src/
│   ├── ArbitraryInstitutions.jl    # 主模块
│   ├── Brain/                       # 认知架构
│   │   ├── Types.jl                 # 信念状态、认知状态
│   │   ├── FactorGraph.jl           # M0/M1 生成模型
│   │   ├── Learning.jl              # 贝叶斯更新、模型选择、内化
│   │   └── ActionSelection.jl       # EFE + 互惠机制
│   ├── World/                       # 环境
│   │   ├── Types.jl                 # 智能体、交互记录
│   │   ├── Physics.jl               # 博弈类型 (PD/SH/Harmony)
│   │   └── Dynamics.jl              # 模拟循环
│   └── Analytics/                   # 分析
│       ├── Convergence.jl           # 涌现指标
│       └── Visualization.jl         # 可视化
├── experiments/
│   ├── run_parameter_sweep.jl       # 并行实验框架
│   ├── exp1_minimal_trigger.jl      # 实验1-8
│   ├── ...
│   ├── validate_literature.jl       # 文献验证
│   └── results/                     # CSV结果
├── docs/
│   ├── conceptual_model.md          # 概念模型图示
│   ├── theoretical_validation.md    # 理论验证
│   └── verification_report.md       # 可靠性验证报告
└── test/
    └── runtests.jl                  # 182个测试
```

---

## 理论贡献 / Theoretical Contributions

### 1. 统一框架
主动推理 + 贝叶斯模型选择 + 内化动力学 = 制度涌现的计算理论

### 2. 新机制发现
- **互惠内化**: γ参数连接信念与行为
- **信号检测权衡**: 先验决定敏感性方向
- **临界质量协调**: 小群体优势的计算解释
- **学习价值效应**: 高功能环境降低制度学习动机 [验证后新增]

### 3. 可测试预测
- **H1**: 悲观者对正向信号更敏感，乐观者对负向信号更敏感
- **H2**: 8-24人群体比64+人群体更容易形成规范
- **H3**: 长期制度"空壳化"——结构存在但行为趋同
- **H4**: 高功能环境抑制制度涌现（非通过信号消失，而是动机降低）

### 4. 实践启示
- 制度设计应**小群体试点 → 逐步扩展**
- 早期小偏差比晚期大干预更有效
- 协调问题(猎鹿)比困境(囚徒)更适合制度解决
- **新**：越有用的制度越难涌现——需要显式设计学习机会

---

## 安装 / Installation

```bash
git clone https://github.com/your-repo/ArbitraryInstitutions.jl.git
cd ArbitraryInstitutions.jl
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

### 运行测试

```bash
julia --project=. -e "using Pkg; Pkg.test()"
# 182 tests passed
```

---

## 依赖 / Dependencies

- [Agents.jl](https://juliadynamics.github.io/Agents.jl/) - 智能体建模
- [RxInfer.jl](https://rxinfer.ml/) - 贝叶斯推理
- [GLMakie.jl](https://docs.makie.org/) - 可视化
- [DataFrames.jl](https://dataframes.juliadata.org/) - 数据分析

---

## 引用 / Citation

```bibtex
@software{arbitrary_institutions_jl,
  title = {Arbitrary-Institutions.jl: Learnability-Based Theory of Institution Emergence},
  year = {2025},
  note = {Demonstrates that institution emergence depends on learnability,
          not functionality, through Active Inference and Bayesian model selection}
}
```

---

## 相关文献 / References

- Friston, K. (2010). The free-energy principle: a unified brain theory?
- Tajfel, H. (1971). Social identity and intergroup relations (MGP)
- Yamagishi, T. (1999). Trust, gullibility, and social intelligence
- Balliet, D. et al. (2014). Ingroup favoritism in cooperation: A meta-analysis
- Da Costa, L. et al. (2020). Active inference on discrete state-spaces

---

## License

MIT License

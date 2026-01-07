# Parameter Exploration Experiments

Systematic parameter exploration for understanding institution emergence dynamics.

## Overview

Total: ~4260 runs (20 repeats per configuration)

| Experiment | Description | Configurations | Runs |
|------------|-------------|----------------|------|
| 1 | Minimal Trigger Conditions | 48 | 960 |
| 2 | Complexity Penalty Sensitivity | 21 | 420 |
| 3 | Action Precision Effect | 15 | 300 |
| 4 | Internalization Dynamics | 36 | 720 |
| 5 | Population Scale Effect | 15 | 300 |
| 6 | Game Type Comparison | 36 | 720 |
| 7 | Prior Belief Effect | 18 | 360 |
| 8 | Long-term Stability | 24 | 480 |

## Quick Start

```bash
# Run all experiments
julia --project=. experiments/run_all_experiments.jl

# Run specific experiments
julia --project=. experiments/run_all_experiments.jl 1 2 6

# Run single experiment
julia --project=. experiments/exp1_minimal_trigger.jl

# Analyze results
julia --project=. -e 'include("experiments/analyze_results.jl"); generate_full_report()'
```

## Experiment Descriptions

### Experiment 1: Minimal Trigger Conditions
**Goal**: Find minimum bias to trigger self-fulfilling prophecy.
- bias: [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
- bias_duration: [25, 50, 100, 200]
- game_type: [PD, StagHunt]

### Experiment 2: Complexity Penalty Sensitivity
**Goal**: How Occam's razor affects model selection.
- complexity_penalty: [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
- bias: [0.05, 0.10, 0.15]

### Experiment 3: Action Precision Effect
**Goal**: Exploration-exploitation trade-off.
- action_precision (β): [0.5, 1.0, 2.0, 5.0, 10.0]
- bias: [0.05, 0.10, 0.15]

### Experiment 4: Internalization Dynamics
**Goal**: How γ update affects feedback loop.
- γ_update_factor: [1.01, 1.05, 1.10, 1.20]
- initial_precision: [0.5, 1.0, 2.0]
- max_precision: [5.0, 10.0, 20.0]

### Experiment 5: Population Scale Effect
**Goal**: Critical mass and scale relationships.
- n_agents: [8, 16, 32, 64, 128]
- bias: [0.05, 0.10, 0.15]

### Experiment 6: Game Type Comparison
**Goal**: How social dilemma structure affects emergence.
- game_type: [PD, StagHunt, Harmony]
- bias: [0.05, 0.10, 0.15, 0.20]
- complexity_penalty: [0.01, 0.05, 0.1]

### Experiment 7: Prior Belief Effect
**Goal**: Impact of initial optimism/pessimism.
- prior: [Pessimistic(1,3), Neutral(1,1), Optimistic(3,1)]
- game_type: [PD, StagHunt]
- bias: [0.05, 0.10, 0.15]

### Experiment 8: Long-term Stability
**Goal**: Do institutions persist or decay?
- total_length: [500, 1000, 2000, 5000]
- bias: [0.10, 0.15, 0.20]
- bias_duration: [100, 200]

## Output

Results are saved to `experiments/results/` as CSV files with timestamps.

Key metrics collected:
- `institutional_adoption_rate` - % agents using M1
- `mean_internalization` - average γ value
- `cooperation_gap` - True vs False label cooperation difference
- `self_fulfilling` - institution emerged AND maintained behavioral difference

## File Structure

```
experiments/
├── README.md                    # This file
├── run_parameter_sweep.jl       # Core experiment framework
├── run_all_experiments.jl       # Master runner script
├── analyze_results.jl           # Analysis and visualization
├── exp1_minimal_trigger.jl      # Experiment 1
├── exp2_complexity_penalty.jl   # Experiment 2
├── exp3_action_precision.jl     # Experiment 3
├── exp4_internalization.jl      # Experiment 4
├── exp5_population_scale.jl     # Experiment 5
├── exp6_game_types.jl           # Experiment 6
├── exp7_prior_beliefs.jl        # Experiment 7
├── exp8_long_term.jl            # Experiment 8
└── results/                     # Output directory (created automatically)
```

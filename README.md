# Goal State Agent

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clean implementation of goal state agents for model-based reinforcement learning.

## Overview

This library implements the **goal state agent** architecture where:

1. An **action policy network** produces actions given the current state
2. A **prediction module** predicts the next state given (state, action)
3. The predicted next state is compared to a **goal state**
4. Gradients flow backward through the prediction module to update the action policy

This is a form of model-based learning where the learned prediction module serves as a differentiable simulator for action policy optimization.

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   State     │────▶│  Action Policy   │────▶│     Action      │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                            ▲                         │
                            │                         ▼
                    ┌───────┴────────┐     ┌─────────────────┐
                    │   Gradients    │◀────│ Prediction      │
                    │   (backprop)   │     │ Module          │
                    └────────────────┘     └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │ Predicted State │
                                           └────────┬────────┘
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │   Goal State    │
                                           │   Comparison    │
                                           └─────────────────┘
```

## Installation

### From GitHub

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/goal-state-agent.git
cd goal-state-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### As a Package

```bash
# Install in development mode
pip install -e .

# Or install directly from GitHub
pip install git+https://github.com/YOUR_USERNAME/goal-state-agent.git
```

## Quick Start

```python
from goal_state_agent import Trainer, TrainingConfig

# Train with default settings
config = TrainingConfig(max_episodes=500)
trainer = Trainer(config=config)
metrics = trainer.train()

# Evaluate
results = trainer.evaluate(num_episodes=10)
print(f"Average episode length: {results['mean_episode_length']}")
```

## Using the CLI

```bash
# Basic training
python -m goal_state_agent.examples.train_cartpole

# With custom hyperparameters
python -m goal_state_agent.examples.train_cartpole --action-lr 0.1 --prediction-lr 0.1

# With Adam optimizer
python -m goal_state_agent.examples.train_cartpole --optimizer adam --action-lr 0.001

# See all options
python -m goal_state_agent.examples.train_cartpole --help
```

## Configuration

All hyperparameters can be set via `TrainingConfig`:

```python
from goal_state_agent import TrainingConfig

config = TrainingConfig(
    # Environment
    env_name="CartPole-v1",
    max_episodes=1000,
    max_steps_per_episode=500,

    # Learning rates
    action_learning_rate=0.5,
    prediction_learning_rate=0.5,
    use_adaptive_lr=True,

    # Inner loop iterations
    action_update_steps=5,
    prediction_update_steps=5,

    # Goal state (for CartPole: angle and angular velocity)
    goal_indices=[2, 3],
    goal_values=[0.0, 0.0],

    # Optimizer: 'builtin', 'sgd', or 'adam'
    optimizer_type="builtin",
)
```

### Preset Configurations

```python
from goal_state_agent import CARTPOLE_FAST, CARTPOLE_ADAM, CARTPOLE_STABLE

# Fast convergence with adaptive LR
trainer = Trainer(config=CARTPOLE_FAST)

# Stable training with Adam
trainer = Trainer(config=CARTPOLE_ADAM)
```

## Architecture Details

### Action Policy
- Input: State (4 dimensions for CartPole)
- Output: Action (1 dimension, thresholded to 0/1)
- Network: Linear layer (4 → 1)

### Prediction Module
- Input: [State, Action] (5 dimensions)
- Output: Next State (4 dimensions)
- Network: Linear layer (5 → 4)

### Gradient Flow

The key mechanism is that gradients from the goal state error flow through the prediction module's weights to update the action policy:

```
Goal State Error
      ↓
∂cost/∂predicted_next_state
      ↓
  W_prediction[action_input_idx, goal_indices]  ← gradient bridge
      ↓
∂cost/∂W_action
      ↓
Update action policy
```

## Extending the Library

### Custom Layers

```python
from goal_state_agent.core.layers import Layer

class MyLayer(Layer):
    def forward(self, x): ...
    def get_pass_through_gradient(self): ...
    def compute_parameter_gradients(self, x): ...
    def apply_gradients(self, pass_through_gradients, cost_gradient, learning_rate): ...
```

### Custom Loss Functions

```python
from goal_state_agent.core.losses import Loss

class MyLoss(Loss):
    def compute(self, y_pred, y_true): ...
    def gradient(self, y_pred, y_true): ...
```

### Goal-Conditioned Policies

To extend to multiple goal states, modify `AgentConfig` to include goal as input:

```python
# Future extension: goal as input to action policy
action_input_dim = state_dim + goal_dim
```

## Visualization

### Training Plots

Generate plots of training metrics:

```python
from goal_state_agent import Trainer, TrainingConfig
from goal_state_agent.visualization import plot_training_metrics

trainer = Trainer(TrainingConfig(max_episodes=100))
metrics = trainer.train()

plot_training_metrics(
    episode_lengths=metrics.episode_lengths,
    goal_errors=metrics.goal_errors,
    prediction_errors=metrics.prediction_errors,
    save_dir="figures",
)
```

### Learning Progression Videos

Record videos at different stages of training to visualize learning:

```python
from goal_state_agent.visualization import record_learning_progression

results = record_learning_progression(
    max_episodes=100,
    record_episodes=[0, 10, 25, 50, 99],  # Episodes to record
    video_folder="videos/learning",
)
```

Or use the CLI:

```bash
# Generate plots only
python -m goal_state_agent.examples.visualize_training --output-dir results

# Record videos of learning progression
python -m goal_state_agent.examples.visualize_training --record-video --output-dir results

# Create a montage of all recorded episodes (requires moviepy)
python -m goal_state_agent.examples.visualize_training --record-video --create-montage
```

## Project Structure

```
goal_state_agent/
├── __init__.py              # Main exports
├── training.py              # Training loop and Trainer class
├── agents/
│   └── goal_state_agent.py  # GoalStateAgent implementation
├── core/
│   ├── layers.py            # Neural network layers
│   └── losses.py            # Loss functions
├── configs/
│   └── config.py            # Configuration dataclasses
├── utils/
│   └── optimizers.py        # SGD, Adam optimizers
├── visualization/
│   ├── plots.py             # Training metric plots
│   └── video.py             # Video recording utilities
├── examples/
│   ├── train_cartpole.py    # Basic training script
│   └── visualize_training.py # Training with visualization
├── requirements.txt
└── pyproject.toml
```

## Results

On CartPole-v1, the agent typically converges within 20-50 episodes:

```
Episode 0: steps=22, avg_steps=22.0, goal_err=0.1697
Episode 10: steps=206, avg_steps=149.3, goal_err=0.0610
Episode 20: steps=500, avg_steps=450.2, goal_err=0.0012
Converged at episode 22
```

## Background

This implementation is based on the model-based learning paradigm where:
- A learned **world model** (prediction module) simulates environment dynamics
- The world model is **differentiable**, allowing gradient-based policy optimization
- The policy is optimized to reach a specified **goal state**

Similar approaches include:
- World Models (Ha & Schmidhuber, 2018)
- Dreamer (Hafner et al., 2019)
- Model-Based Policy Optimization (Janner et al., 2019)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

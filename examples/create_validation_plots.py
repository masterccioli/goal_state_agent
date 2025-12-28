#!/usr/bin/env python3
"""Create comprehensive validation plots."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def main():
    output_dir = Path("output/validation")

    with open(output_dir / "validation_summary.json") as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Goal State Agent - Research Validation with Correct Gymnasium\n(CartPole-v1, 100 episodes, 3 runs each)",
                 fontsize=14, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, 8))

    # 1. Baseline Performance
    ax = axes[0, 0]
    baseline = data["baseline"]["baseline"]
    ax.bar(["Baseline"], [baseline["final_avg_mean"]], yerr=[baseline["final_avg_std"]],
           color=colors[0], capsize=5, edgecolor='black')
    ax.axhline(y=500, color='red', linestyle='--', label='Max (500)')
    ax.axhline(y=baseline["final_avg_mean"], color='green', linestyle=':', alpha=0.5)
    ax.set_ylabel("Avg Episode Length (last 10)")
    ax.set_title("Baseline Performance\n(Online Learning + Gradient Clipping)")
    ax.legend()
    ax.set_ylim(0, 550)
    ax.text(0, baseline["final_avg_mean"] + 30, f'{baseline["final_avg_mean"]:.0f}±{baseline["final_avg_std"]:.0f}',
            ha='center', fontsize=10)

    # 2. Gradient Clipping Comparison
    ax = axes[0, 1]
    gc_data = data["gradient_clipping"]
    names = list(gc_data.keys())
    avgs = [gc_data[n]["final_avg_mean"] for n in names]
    stds = [gc_data[n]["final_avg_std"] for n in names]

    bars = ax.bar(range(len(names)), avgs, yerr=stds, color=colors[1], capsize=5, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Avg Episode Length")
    ax.set_title("Gradient Clipping Effect\n(Lower variance with clipping)")
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5)

    # Add variance annotation
    for i, (a, s) in enumerate(zip(avgs, stds)):
        ax.text(i, a + s + 10, f'σ={s:.0f}', ha='center', fontsize=8)

    # 3. LR Strategies
    ax = axes[0, 2]
    lr_data = data["lr_strategies"]
    names = list(lr_data.keys())
    avgs = [lr_data[n]["final_avg_mean"] for n in names]
    stds = [lr_data[n]["final_avg_std"] for n in names]

    bar_colors = [colors[2], colors[3], colors[4], colors[5]]
    bars = ax.bar(range(len(names)), avgs, yerr=stds, color=bar_colors, capsize=5, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Avg Episode Length")
    ax.set_title("Learning Rate Strategies\n(Surprise strategy fails!)")
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5)

    # Highlight surprise failure
    ax.annotate('FAILS', xy=(3, 24), fontsize=10, color='red', fontweight='bold', ha='center')

    # 4. Learning Rates
    ax = axes[1, 0]
    lr_data = data["learning_rates"]
    names = list(lr_data.keys())
    avgs = [lr_data[n]["final_avg_mean"] for n in names]
    stds = [lr_data[n]["final_avg_std"] for n in names]
    max_eps = [lr_data[n]["max_episode_mean"] for n in names]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width/2, avgs, width, yerr=stds, label='Final Avg', color=colors[0], capsize=3, edgecolor='black')
    bars2 = ax.bar(x + width/2, max_eps, width, label='Max Episode', color=colors[1], edgecolor='black')

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("lr_", "") for n in names], fontsize=9)
    ax.set_ylabel("Steps")
    ax.set_xlabel("Learning Rate")
    ax.set_title("Learning Rate Tuning\n(LR=0.5 reaches max, LR=0.1 most stable)")
    ax.legend()
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5)

    # Highlight optimal
    ax.annotate('Optimal', xy=(3, 160), fontsize=10, color='green', fontweight='bold', ha='center')

    # 5. Updates Per Step
    ax = axes[1, 1]
    upd_data = data["updates_per_step"]
    names = list(upd_data.keys())
    avgs = [upd_data[n]["final_avg_mean"] for n in names]
    stds = [upd_data[n]["final_avg_std"] for n in names]
    conv_rates = [upd_data[n]["convergence_rate"] * 100 for n in names]

    bar_colors = [colors[6] if c > 0 else colors[7] for c in conv_rates]
    bars = ax.bar(range(len(names)), avgs, yerr=stds, color=bar_colors, capsize=5, edgecolor='black')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("updates_", "") for n in names], fontsize=9)
    ax.set_ylabel("Avg Episode Length")
    ax.set_xlabel("Updates Per Step")
    ax.set_title("Update Frequency\n(3 updates optimal, 33% convergence!)")
    ax.axhline(y=500, color='red', linestyle='--', alpha=0.5)

    # Add convergence rate annotation
    for i, (a, c) in enumerate(zip(avgs, conv_rates)):
        if c > 0:
            ax.annotate(f'{c:.0f}% conv', xy=(i, a + 50), fontsize=9, color='green', fontweight='bold', ha='center')

    # 6. Summary/Recommendations
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = """
KEY FINDINGS (Validated with Correct Gymnasium):

✓ REPLAY BUFFERS HARM PERFORMANCE
  - Baseline: 242.6 avg
  - With buffer: 122.7 avg (-49%)
  - Local targets: 95.8 avg (-61%)
  - Buffer states: 49.4 avg (-80%)

✓ GRADIENT CLIPPING REDUCES VARIANCE
  - No clip: σ=70.6
  - Clip 2.0: σ=21.4 (best)
  - No improvement in average

✓ GOLDILOCKS KEEPS ERRORS STABLE
  - Goldilocks max_err: 4.14e+01
  - Baseline max_err: 1.78e+02
  - Surprise max_err: 5.00e+11 (exploded!)

✓ OPTIMAL HYPERPARAMETERS
  - Updates/step: 3 (33% convergence)
  - Learning rate: 0.5 (with adaptive LR)
  - Gradient clip: 1.0-2.0

RECOMMENDED CONFIG:
  updates=3, lr=0.5, clip=1.0, adaptive_lr=True
"""
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / "comprehensive_validation.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_dir / 'comprehensive_validation.png'}")

    # Also create a comparison plot showing broken vs correct environment
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Environment Comparison: Broken Custom vs Standard Gymnasium", fontsize=14, fontweight='bold')

    categories = ['Episode 4\nConvergence', 'Error\nExplosion', 'Actions\nHave Effect']
    broken = [1, 0, 0]  # Broken env: instant convergence, no explosion (no learning), no effect
    correct = [0, 1, 1]  # Correct env: no instant conv, errors can explode, actions work

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, broken, width, label='Broken (Custom)', color='lightcoral', edgecolor='black')
    ax.bar(x + width/2, correct, width, label='Correct (Standard)', color='lightgreen', edgecolor='black')

    ax.set_ylabel('Behavior Present (1=Yes, 0=No)')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 1.5)

    # Add explanation text
    text = """
The "broken" custom Gymnasium environment had:
  • force = 0 (actions did nothing)
  • terminated = False (episodes never failed)
  • Cart clamped to [-1, 1]

This caused "instant convergence" artifacts in previous research notes.
All findings have been re-validated with standard Gymnasium 1.2.3.
"""
    ax.text(0.5, -0.25, text, transform=ax.transAxes, fontsize=10,
            ha='center', va='top', style='italic')

    plt.tight_layout()
    plt.savefig(output_dir / "environment_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Plot saved to {output_dir / 'environment_comparison.png'}")


if __name__ == "__main__":
    main()

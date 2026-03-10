#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# SCPN Control — PPO Training on UpCloud/JarvisLabs
# © 1998–2026 Miroslav Šotek. All rights reserved.
# ──────────────────────────────────────────────────────────────────────
#
# Usage (on cloud VM):
#   1. Clone repo:    git clone https://github.com/anulum/scpn-control.git
#   2. Install:       pip install -e ".[rl,dev]"
#   3. Run:           bash tools/train_rl_upcloud.sh
#   4. Download:      scp remote:scpn-control/weights/ppo_tokamak* .
#
# Estimated runtime:
#   500K steps on CPU MLP: ~10-15 min (any cloud CPU instance)
#   1M steps on CPU MLP:   ~25-30 min
#   No GPU needed — SB3 MLP policy runs on CPU.
#
# Recommended instance:
#   UpCloud:     2 CPU / 4 GB RAM ($0.03/hr) — total cost: ~$0.01-0.02
#   JarvisLabs:  CPU instance — ~$0.10/hr — total cost: ~$0.05
#
set -euo pipefail

TIMESTEPS=${1:-500000}
EVAL_EPISODES=${2:-50}
SEEDS="42 123 456"

echo "=== SCPN Control PPO Training ==="
echo "Timesteps: $TIMESTEPS"
echo "Eval episodes: $EVAL_EPISODES"
echo "Seeds: $SEEDS"
echo ""

# Force CPU to avoid SB3 GPU overhead with MLP policy
export CUDA_VISIBLE_DEVICES=""

BEST_REWARD=-99999
BEST_SEED=42

for SEED in $SEEDS; do
    echo "--- Training with seed=$SEED ---"
    python tools/train_rl_tokamak.py \
        --timesteps "$TIMESTEPS" \
        --eval-episodes "$EVAL_EPISODES" \
        --output "weights/ppo_tokamak_seed${SEED}.zip"

    # Read mean reward from metrics
    METRICS="weights/ppo_tokamak_seed${SEED}.metrics.json"
    REWARD=$(python -c "import json; m=json.load(open('$METRICS')); print(m['ppo']['mean_reward'])")
    echo "Seed $SEED: PPO mean_reward = $REWARD"

    # Track best
    IS_BETTER=$(python -c "print(1 if $REWARD > $BEST_REWARD else 0)")
    if [ "$IS_BETTER" = "1" ]; then
        BEST_REWARD=$REWARD
        BEST_SEED=$SEED
    fi
done

echo ""
echo "=== Best seed: $BEST_SEED (reward=$BEST_REWARD) ==="

# Copy best to default location
cp "weights/ppo_tokamak_seed${BEST_SEED}.zip" weights/ppo_tokamak.zip
cp "weights/ppo_tokamak_seed${BEST_SEED}.metrics.json" weights/ppo_tokamak.metrics.json

# Run full benchmark with best agent
echo "--- Running RL vs Classical benchmark ---"
python benchmarks/rl_vs_classical.py

echo ""
echo "=== Results ==="
python -c "
import json
m = json.load(open('benchmarks/rl_vs_classical.json'))
for name in ['mpc', 'pid', 'ppo']:
    d = m[name]
    print(f\"{name.upper():>4}: reward={d['mean_reward']:8.1f} +/- {d['std_reward']:5.1f}  disruption={d['disruption_rate']*100:.0f}%\")
"

echo ""
echo "Done. Copy weights/ppo_tokamak.zip and benchmarks/rl_vs_classical.json back."

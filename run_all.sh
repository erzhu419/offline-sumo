#!/bin/bash
# ============================================================================
# run_all.sh — Offline-to-Online RL experiments for SUMO bus holding
# ============================================================================
#
# Runs all experiments needed for the paper:
#   1. Offline-only evaluation (no fine-tune)
#   2. Pure online SAC (lower bound baseline)
#   3. WSRL (policy-dominant: pretrain + online SAC)
#   4. RLPD (data-dominant: random init + 50% offline mixing)
#   5. RLPD ablation: offline_ratio = {0.25, 0.75}
#
# Each method runs with 3 seeds (42, 123, 456).
# Designed for server deployment — adjust CONDA_ENV and N_EPOCHS as needed.
#
# Usage:
#   cd /home/erzhu419/mine_code/offline-sumo
#   bash run_all.sh              # run all
#   bash run_all.sh wsrl         # run only WSRL
#   bash run_all.sh rlpd         # run only RLPD
#   bash run_all.sh online       # run only online baseline
#   bash run_all.sh eval         # run only offline eval
#   bash run_all.sh ablation     # run only RLPD ablation
# ============================================================================

set -e

CONDA_ENV="LSTM-RL"
N_EPOCHS=300
DEVICE="cpu"
SEEDS=(42 123 456)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/experiment_output/logs"
mkdir -p "$LOG_DIR"

TARGET="${1:-all}"

echo "============================================"
echo " Offline-to-Online RL Experiments"
echo " epochs=$N_EPOCHS, seeds=${SEEDS[*]}, device=$DEVICE"
echo " target=$TARGET"
echo "============================================"

# ── 1. Offline-only evaluation ──────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "eval" ]]; then
    echo ""
    echo "[1/5] Offline-only evaluation (10 episodes)..."
    conda run -n $CONDA_ENV python eval_offline.py \
        --n_eval 10 --device $DEVICE \
        2>&1 | tee "$LOG_DIR/eval_offline.log"
fi

# ── 2. Pure online SAC ──────────────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "online" ]]; then
    echo ""
    echo "[2/5] Pure online SAC..."
    for seed in "${SEEDS[@]}"; do
        echo "  → seed=$seed"
        conda run -n $CONDA_ENV python train_online_only.py \
            --seed $seed --n_epochs $N_EPOCHS --device $DEVICE \
            2>&1 | tee "$LOG_DIR/online_seed${seed}.log"
    done
fi

# ── 3. WSRL (policy-dominant) ───────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "wsrl" ]]; then
    echo ""
    echo "[3/5] WSRL (warm-start + online SAC)..."
    for seed in "${SEEDS[@]}"; do
        echo "  → seed=$seed"
        conda run -n $CONDA_ENV python train_wsrl.py \
            --seed $seed --n_epochs $N_EPOCHS --device $DEVICE \
            2>&1 | tee "$LOG_DIR/wsrl_seed${seed}.log"
    done
fi

# ── 4. RLPD (data-dominant, offline_ratio=0.5) ─────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "rlpd" ]]; then
    echo ""
    echo "[4/5] RLPD (random init + 50% offline mixing)..."
    for seed in "${SEEDS[@]}"; do
        echo "  → seed=$seed"
        conda run -n $CONDA_ENV python train_rlpd.py \
            --seed $seed --n_epochs $N_EPOCHS --device $DEVICE \
            --offline_ratio 0.5 \
            2>&1 | tee "$LOG_DIR/rlpd_seed${seed}.log"
    done
fi

# ── 5. RLPD ablation: offline_ratio sweep ──────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "ablation" ]]; then
    echo ""
    echo "[5/5] RLPD ablation (offline_ratio sweep)..."
    for ratio in 0.25 0.75; do
        for seed in "${SEEDS[@]}"; do
            echo "  → offline_ratio=$ratio, seed=$seed"
            conda run -n $CONDA_ENV python train_rlpd.py \
                --seed $seed --n_epochs $N_EPOCHS --device $DEVICE \
                --offline_ratio $ratio \
                2>&1 | tee "$LOG_DIR/rlpd_ratio${ratio}_seed${seed}.log"
        done
    done
fi

echo ""
echo "============================================"
echo " All experiments complete!"
echo " Results in: $SCRIPT_DIR/experiment_output/"
echo " Logs in:    $LOG_DIR/"
echo "============================================"

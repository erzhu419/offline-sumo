#!/bin/bash
# ============================================================================
# package.sh — Package offline-sumo for server deployment
# ============================================================================
#
# Creates a tarball containing everything needed to run all experiments.
# Structure in tarball:
#   offline-sumo-pkg/
#   ├── offline-sumo/              (code + data + pretrained model)
#   └── sumo-rl/                   (SUMO network files)
#       └── _standalone_f543609/
#           └── SUMO_ruiguang/
#               └── online_control/
#
# On server, extract and run:
#   tar xzf offline-sumo-pkg.tar.gz
#   cd offline-sumo-pkg/offline-sumo
#   python auto_run.py
# ============================================================================

set -e

HOME_DIR="/home/erzhu419/mine_code"
PKG_NAME="offline-sumo-pkg"
PKG_DIR="/tmp/$PKG_NAME"
OUT_TAR="/tmp/${PKG_NAME}.tar.gz"

echo "Packaging offline-sumo for server deployment..."
rm -rf "$PKG_DIR" "$OUT_TAR"

# ── 1. Copy code (exclude experiment outputs, cache, git) ───────────────────
echo "  [1/4] Copying offline-sumo code..."
mkdir -p "$PKG_DIR/offline-sumo"
rsync -a \
    --exclude 'experiment_output' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude '*.pyc' \
    --exclude 'data/datasets_v2' \
    "$HOME_DIR/offline-sumo/" "$PKG_DIR/offline-sumo/"

# ── 2. Copy offline dataset (dereference symlink) ───────────────────────────
echo "  [2/4] Copying offline dataset (245 MB)..."
mkdir -p "$PKG_DIR/offline-sumo/data/datasets_v2"
cp "$HOME_DIR/offline-sumo/data/datasets_v2/merged_all_v2.h5" \
   "$PKG_DIR/offline-sumo/data/datasets_v2/"

# ── 3. Copy SUMO network (exclude large output files) ──────────────────────
echo "  [3/4] Copying SUMO network files..."
mkdir -p "$PKG_DIR/sumo-rl/_standalone_f543609/SUMO_ruiguang"
rsync -a \
    --exclude 'output_file/PreSimFcd.xml' \
    --exclude '__pycache__' \
    "$HOME_DIR/sumo-rl/_standalone_f543609/SUMO_ruiguang/online_control" \
    "$PKG_DIR/sumo-rl/_standalone_f543609/SUMO_ruiguang/"

# Ensure output_file dir exists (SUMO needs it to write to)
mkdir -p "$PKG_DIR/sumo-rl/_standalone_f543609/SUMO_ruiguang/online_control/output_file"

# ── 4. Create README for server deployment ─────────────────────────────────
cat > "$PKG_DIR/README_SERVER.md" <<'EOF'
# Offline-to-Online RL Experiments — Server Deployment

## Quick Start (one command)

```bash
cd offline-sumo && python auto_run.py
```

That's it. Automatically:
- Detects free GPU memory on each GPU
- Dispatches 16 experiments in parallel across available slots
- Falls back to CPU if no GPU memory available
- Runs summary analysis when done

## Requirements
- Python 3.8+ with PyTorch, numpy, h5py, matplotlib
- SUMO (libsumo) installed and importable: `python -c "import libsumo"`

## Experiments
- 1 offline eval
- 3 × Online SAC (baseline)
- 3 × WSRL (policy-dominant)
- 3 × RLPD (offline_ratio=0.5, standard)
- 3 × RLPD (offline_ratio=0.25, ablation)
- 3 × RLPD (offline_ratio=0.75, ablation)

## Options

```bash
# Test with fewer epochs
python auto_run.py --n_epochs 50

# Run only specific methods
python auto_run.py --only wsrl,rlpd

# Use specific conda env
python auto_run.py --python "conda run -n myenv python"

# Override memory estimate per job (MB)
python auto_run.py --mem_per_job 3000

# Dry run (print plan only)
python auto_run.py --dry_run
```

## Results

After completion, check:
- `offline-sumo/experiment_output/auto_run_logs/` — per-experiment logs
- `offline-sumo/experiment_output/summary_table.csv` — aggregated results
- `offline-sumo/experiment_output/learning_curves.csv` — for plotting
EOF

# ── 5. Pack tarball ─────────────────────────────────────────────────────────
echo "  [4/4] Creating tarball..."
cd /tmp
tar czf "$OUT_TAR" "$PKG_NAME/"

SIZE=$(du -sh "$OUT_TAR" | cut -f1)
echo ""
echo "============================================"
echo " Package ready: $OUT_TAR  ($SIZE)"
echo ""
echo " Upload to server, then:"
echo "   tar xzf ${PKG_NAME}.tar.gz"
echo "   cd ${PKG_NAME}/offline-sumo"
echo "   python auto_run.py"
echo "============================================"

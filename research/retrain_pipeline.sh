#!/bin/bash
# Full retrain pipeline — run all validation steps in correct order.
#
# Usage:
#   bash research/retrain_pipeline.sh            # dry run: stops before deploy
#   bash research/retrain_pipeline.sh --deploy   # actually overwrite production artifacts
#
# Each step will halt on failure. If a step fails, fix the problem before
# proceeding manually. Never skip a step by commenting it out.

set -e

DEPLOY=false
if [ "$1" = "--deploy" ]; then
    DEPLOY=true
fi

# Force UTF-8 output on Windows (cp950 breaks Unicode prints)
export PYTHONIOENCODING=utf-8

echo "============================================================"
echo "RETRAIN PIPELINE — $(date)"
echo "Mode: $([ "$DEPLOY" = true ] && echo 'DEPLOY' || echo 'DRY RUN')"
echo "============================================================"

# ─────────────────────────────────────────────────────────────
# Step 1: Sample size check
# ─────────────────────────────────────────────────────────────
echo ""
echo "[1/9] Sample size check"
python -c "
import pandas as pd
df = pd.read_parquet('research/dual_model/.cache/features_all.parquet')
n = len(df)
print(f'  Current cache: {n} bars ({df.index[0].date()} -> {df.index[-1].date()})')
if n < 5500:
    print(f'  WARNING: only {n} bars. Recommended >= 5500 for retraining with more features.')
    print(f'  You can continue but sample:feature ratio will be tight.')
else:
    print(f'  OK: {n} bars is sufficient.')
"

# ─────────────────────────────────────────────────────────────
# Step 2: Train/serve drift check
# ─────────────────────────────────────────────────────────────
echo ""
echo "[2/9] Train/serve drift check"
python research/train_serve_diff.py --rebuild
if [ $? -ne 0 ]; then
    echo "  FAIL: drift detected. Fix data source issues before retraining."
    exit 1
fi

# ─────────────────────────────────────────────────────────────
# Step 3: Ablation study (forward feature selection)
# ─────────────────────────────────────────────────────────────
echo ""
echo "[3/9] Ablation study — baseline + candidate feature evaluation"
python research/ablation_study.py
if [ $? -ne 0 ]; then
    echo "  FAIL: ablation study failed."
    exit 1
fi
echo "  Review research/results/ablation_study.json before continuing."
echo "  Only features with verdict='KEEP' will be used for retrain."

# ─────────────────────────────────────────────────────────────
# Step 4: Permutation test on the ablation feature set
# ─────────────────────────────────────────────────────────────
echo ""
echo "[4/9] Permutation test — verify IC is not overfitting"
python research/permutation_test.py --use-ablation --n 50
if [ $? -ne 0 ]; then
    echo "  FAIL: permutation test failed."
    exit 1
fi
echo "  Check research/results/permutation_test.json"
echo "  Required: p_auc < 0.05 AND p_ic < 0.05"

# ─────────────────────────────────────────────────────────────
# Step 5: Backup current production models
# ─────────────────────────────────────────────────────────────
echo ""
echo "[5/9] Backup current production artifacts"
BACKUP_DIR="indicator/model_artifacts/dual_model_backup_$(date +%Y%m%d)"
if [ -d "$BACKUP_DIR" ]; then
    echo "  Backup already exists at $BACKUP_DIR (skipping overwrite)"
else
    cp -r indicator/model_artifacts/dual_model "$BACKUP_DIR"
    echo "  Backup: $BACKUP_DIR"
fi

# ─────────────────────────────────────────────────────────────
# Step 6: Feature importance pre-snapshot
# ─────────────────────────────────────────────────────────────
echo ""
echo "[6/9] Feature importance snapshot (pre-retrain)"
python research/feature_importance_tracker.py snapshot

if [ "$DEPLOY" = false ]; then
    echo ""
    echo "============================================================"
    echo "DRY RUN complete. Review the artifacts:"
    echo "  - research/results/ablation_study.json"
    echo "  - research/results/permutation_test.json"
    echo "  - research/results/train_serve_diff.json"
    echo ""
    echo "If everything looks good, re-run with --deploy to actually retrain."
    echo "============================================================"
    exit 0
fi

# ─────────────────────────────────────────────────────────────
# Step 7: Retrain and export production models
# ─────────────────────────────────────────────────────────────
echo ""
echo "[7/9] Retraining and exporting production models"
python research/deploy_new_models.py
if [ $? -ne 0 ]; then
    echo "  FAIL: model export failed. Restoring backup."
    cp -r "$BACKUP_DIR"/* indicator/model_artifacts/dual_model/
    exit 1
fi

# ─────────────────────────────────────────────────────────────
# Step 8: Calibration check on new model (needs prod predictions)
# ─────────────────────────────────────────────────────────────
echo ""
echo "[8/9] Calibration check (NOTE: uses existing indicator_history,"
echo "      new model's calibration will only be known after 1-2 weeks of live predictions)"
python research/calibration_check.py || echo "  (calibration check failed but not fatal)"

# ─────────────────────────────────────────────────────────────
# Step 9: Feature importance post-snapshot + diff
# ─────────────────────────────────────────────────────────────
echo ""
echo "[9/9] Feature importance post-snapshot + diff"
python research/feature_importance_tracker.py snapshot
python research/feature_importance_tracker.py diff --from previous --to latest

echo ""
echo "============================================================"
echo "RETRAIN COMPLETE"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Review feature importance diff above — any large shifts are red flags"
echo "  2. Restart indicator service to load new model:"
echo "     railway restart indicator (or equivalent)"
echo "  3. Monitor first 24h of live predictions via /perf command"
echo "  4. If anything looks wrong, restore backup:"
echo "     cp -r $BACKUP_DIR/* indicator/model_artifacts/dual_model/"
echo "  5. Commit and push:"
echo "     git add indicator/model_artifacts/dual_model/"
echo "     git commit -m 'Retrain dual model (ablation-validated)'"
echo "     git push origin main"
echo ""

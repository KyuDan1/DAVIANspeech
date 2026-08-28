#!/usr/bin/env bash
# Score an eval set built by scripts/build_eval_*.py with the full pipeline.
#
# The pipeline wants one directory of audio named by ID; eval sets instead carry
# a source_path per row. Materialise that as a symlink farm, run, then report.
#
#   scripts/run_eval_set.sh <eval_set_dir> [extra pipeline.py args...]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ENV_FILE:-$HERE/../env.sh}"

SET_DIR="$(cd "$1" && pwd)"; shift
WORK="${WORK:-$SET_DIR/run}"
mkdir -p "$WORK/audio"

PYTHONNOUSERSITE=1 "$PY" - "$SET_DIR" "$WORK" <<'PYEOF'
import sys, pandas as pd
from pathlib import Path
set_dir, work = Path(sys.argv[1]), Path(sys.argv[2])
truth = pd.read_csv(set_dir / "truth.csv", dtype={"ID": str})
audio = work / "audio"
for old in audio.glob("*"):
    old.unlink()
for row in truth.itertuples():
    src = Path(row.source_path)
    (audio / f"{row.ID}{src.suffix}").symlink_to(src)
print(f"linked {len(truth)} clips into {audio}")
PYEOF

PYTHONNOUSERSITE=1 PATH="$CONDA_ROOT/envs/$ENV_NAME/bin:$PATH" \
"$PY" "$HERE/src/pipeline.py" \
    --test-dir "$WORK/audio" \
    --sample-submission "$SET_DIR/sample_submission.csv" \
    --output "$WORK/submission.csv" \
    --panns-dir "${PANNS_DIR:-$HERE/../models/panns}" \
    --xlsr-dir "${XLSR_DIR:-$HERE/../models/xls-r-2b-anti-deepfake}" \
    --artifactnet-dir "${ARTIFACTNET_DIR:-$HERE/../models/artifactnet}" \
    --htdemucs-repo "${HTDEMUCS_REPO:-$HERE/../baseline/model/htdemucs}" \
    "$@"

PYTHONNOUSERSITE=1 "$PY" "$HERE/src/evaluate_diagnostic.py" \
    "$WORK/submission.csv" "$SET_DIR/truth.csv"

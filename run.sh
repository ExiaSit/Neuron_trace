#!/usr/bin/env bash
set -euo pipefail
METHOD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec /home/pzy/anaconda3/envs/hb_seg/bin/python "$METHOD_DIR/run_pipeline.py" "$@"

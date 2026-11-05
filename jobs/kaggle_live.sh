#!/usr/bin/env bash
# Follow the live log of a Kaggle kernel. Usage: jobs/kaggle_live.sh <owner/kernel-slug>
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: $0 <owner/kernel-slug>" >&2; exit 2; }
exec kaggle kernels logs "$1" --follow

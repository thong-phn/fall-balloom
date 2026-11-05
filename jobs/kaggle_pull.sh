#!/usr/bin/env bash
set -euo pipefail

usage="Usage: $0 <owner/kernel-slug> [--download log|artifact]"

if (( $# != 1 && $# != 3 )); then
  printf '%s\n' "$usage" >&2
  exit 2
fi

kernel="$1"
option="${2:-}"
target="${3:-}"
if [[ -n "$option" && "$option" != "--download" ]]; then
  printf 'Unknown option: %s\n%s\n' "$option" "$usage" >&2
  exit 2
fi

kernel_name="${kernel##*/}"

case "$target" in
  "")
    kaggle kernels logs "$kernel" --follow
    ;;
  log)
    out_dir="./.kaggle-logs"
    timestamp="$(date '+%Y%m%d-%H%M%S')"
    log_file="$out_dir/$kernel_name-$timestamp.log"
    mkdir -p "$out_dir"
    printf 'Following %s; saving log to %s\n' "$kernel" "$log_file" >&2
    kaggle kernels logs "$kernel" --follow 2>&1 | tee "$log_file"
    ;;
  artifact)
    out_dir="./.kaggle-artifacts/$kernel_name"
    mkdir -p "$out_dir"
    printf 'Downloading artifacts of %s to %s\n' "$kernel" "$out_dir" >&2
    kaggle kernels output "$kernel" -p "$out_dir"
    ;;
  *)
    printf 'Unknown download target: %s\n%s\n' "$target" "$usage" >&2
    exit 2
    ;;
esac
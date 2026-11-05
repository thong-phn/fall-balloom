#!/usr/bin/env bash
set -euo pipefail

usage="Usage: $0 <owner/kernel-slug> [--download log|logonly|artifact] [--version N] [--path DIR]
  no --download   follow the live log
  log             follow the log and save it
  logonly         save the log without following
  artifact        download the output files
  --version N     kernel version (default: latest)
  --path DIR      folder for the log file or the artifacts"

if (( $# < 1 )); then
  printf '%s\n' "$usage" >&2
  exit 2
fi

kernel="$1"
shift
target="" version="" path=""
while (( $# )); do
  (( $# >= 2 )) || { printf 'Missing value for %s\n%s\n' "$1" "$usage" >&2; exit 2; }
  case "$1" in
    --download) target="$2" ;;
    --version)  version="$2" ;;
    --path)     path="$2" ;;
    *) printf 'Unknown option: %s\n%s\n' "$1" "$usage" >&2; exit 2 ;;
  esac
  shift 2
done

kernel_name="${kernel##*/}"
ref="$kernel${version:+/$version}"   # <owner>/<slug>[/<version>]
tag="$kernel_name${version:+-v$version}"

case "$target" in
  "")
    kaggle kernels logs "$ref" --follow
    ;;
  log|logonly)
    out_dir="${path:-./.kaggle-logs}"
    log_file="$out_dir/$tag-$(date '+%Y%m%d-%H%M%S').log"
    mkdir -p "$out_dir"
    follow=(); [[ "$target" == log ]] && follow=(--follow)
    printf 'Saving log of %s to %s\n' "$ref" "$log_file" >&2
    kaggle kernels logs "$ref" "${follow[@]}" 2>&1 | tee "$log_file"
    ;;
  artifact)
    out_dir="${path:-./.kaggle-artifacts/$tag}"
    mkdir -p "$out_dir"
    printf 'Downloading artifacts of %s to %s\n' "$ref" "$out_dir" >&2
    # Download to a temp folder, add the date and time to each *.log name, then copy.
    # This leaves logs from earlier pulls in $out_dir untouched.
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    kaggle kernels output "$ref" -p "$tmp"
    stamp="$(date '+%Y%m%d-%H%M%S')"
    find "$tmp" -type f -name '*.log' | while read -r f; do mv "$f" "${f%.log}-$stamp.log"; done
    cp -r "$tmp"/. "$out_dir"/
    ;;
  *)
    printf 'Unknown download target: %s\n%s\n' "$target" "$usage" >&2
    exit 2
    ;;
esac

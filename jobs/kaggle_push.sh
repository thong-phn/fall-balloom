#!/usr/bin/env bash
# Usage: jobs/kaggle_push.sh <owner/kernel-name> [path to folder]   (folder defaults to jobs/kaggle)
set -euo pipefail
[[ $# -eq 1 || $# -eq 2 ]] || { echo "Usage: $0 <owner/kernel-name> [path to folder]" >&2; exit 1; }
KERNEL="$1"
FOLDER="$(dirname "${BASH_SOURCE[0]}")/kaggle"  # default folder
[[ $# -eq 2 ]] && FOLDER="$2"

# Push a copy with the chosen id and title. The Kaggle CLI takes the name from kernel-metadata.json.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp -r "$FOLDER"/. "$TMP"/
sed -i -e "s#\"id\": *\"[^\"]*\"#\"id\": \"$KERNEL\"#" -e "s#\"title\": *\"[^\"]*\"#\"title\": \"${KERNEL#*/}\"#" "$TMP/kernel-metadata.json"

kaggle kernels push -p "$TMP"
kaggle kernels status "$KERNEL"

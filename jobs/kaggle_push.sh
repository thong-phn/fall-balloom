#!/usr/bin/env bash
# Usage: jobs/kaggle_push.sh <path to folder> <owner/kernel-name>
set -euo pipefail
[[ $# -eq 2 ]] || { echo "Usage: $0 <path to folder> <owner/kernel-name>" >&2; exit 1; }
FOLDER="$1"
KERNEL="$2"

# Push a copy with the chosen id and title. The Kaggle CLI takes the name from kernel-metadata.json.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp -r "$FOLDER"/. "$TMP"/
sed -i -e "s#\"id\": *\"[^\"]*\"#\"id\": \"$KERNEL\"#" -e "s#\"title\": *\"[^\"]*\"#\"title\": \"${KERNEL#*/}\"#" "$TMP/kernel-metadata.json"

kaggle kernels push -p "$TMP"
kaggle kernels status "$KERNEL"

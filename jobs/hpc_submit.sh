#!/usr/bin/env bash
# Sync the current fall-balloom source to Vulcan, then submit the BioSAR job.
# Configure VULCAN_HOST and VULCAN_PROJECT_DIR in the environment or in
# jobs/vulcan.env (copy jobs/vulcan.env.example there).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${VULCAN_ENV_FILE:-$SCRIPT_DIR/vulcan.env}"
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_FILE"
fi

: "${VULCAN_HOST:?Set VULCAN_HOST (an SSH host alias or user@host)}"
: "${VULCAN_PROJECT_DIR:?Set VULCAN_PROJECT_DIR to the remote project directory}"
VULCAN_ACCOUNT="${VULCAN_ACCOUNT:-def-sartmu}"
VULCAN_MODEL="${VULCAN_MODEL:-attn-cnn}"
VULCAN_DATA_DIR="${VULCAN_DATA_DIR:-$VULCAN_PROJECT_DIR/datasets/BioSARDdBScene}"
VULCAN_VENV_DIR="${VULCAN_VENV_DIR:-$VULCAN_PROJECT_DIR/.venv}"
VULCAN_POLL_SECONDS="${VULCAN_POLL_SECONDS:-30}"
VULCAN_LOG_TAIL_LINES="${VULCAN_LOG_TAIL_LINES:-80}"

command -v ssh >/dev/null || { echo "ssh is required" >&2; exit 1; }
command -v rsync >/dev/null || { echo "rsync is required" >&2; exit 1; }

printf 'Syncing %s to %s:%s\n' "$PROJECT_DIR" "$VULCAN_HOST" "$VULCAN_PROJECT_DIR"
remote_dir_q="$(printf '%q' "$VULCAN_PROJECT_DIR")"
ssh -o BatchMode=yes -o ConnectTimeout=20 "$VULCAN_HOST" \
    "mkdir -p $remote_dir_q/utils $remote_dir_q/jobs"

# Sync only code and the job definition. Datasets, environments, and model
# outputs remain on the cluster and are never included in this transfer.
(cd "$PROJECT_DIR" && rsync -azR -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' \
    binary-biosar-full.py requirements.txt utils/ jobs/biosar-full.sbatch \
    "$VULCAN_HOST:$VULCAN_PROJECT_DIR/")

printf 'Submitting BioSAR job on %s (model=%s)\n' "$VULCAN_HOST" "$VULCAN_MODEL"
data_dir_q="$(printf '%q' "$VULCAN_DATA_DIR")"
venv_dir_q="$(printf '%q' "$VULCAN_VENV_DIR")"
model_q="$(printf '%q' "$VULCAN_MODEL")"
account_q="$(printf '%q' "$VULCAN_ACCOUNT")"
submission=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$VULCAN_HOST" \
    "cd $remote_dir_q && sbatch --parsable --account=$account_q --export=ALL,DATA_DIR=$data_dir_q,VENV_DIR=$venv_dir_q,MODEL=$model_q jobs/biosar-full.sbatch")
job_id="${submission%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
    echo "Could not parse Slurm job ID from submission response: $submission" >&2
    exit 1
fi
printf 'Submitted job %s\n' "$job_id"

fetch_and_show_logs() {
    local log_dir="$PROJECT_DIR/jobs/vulcan-logs"
    local name remote_log local_log
    mkdir -p "$log_dir"
    for name in out err; do
        remote_log="$VULCAN_PROJECT_DIR/jobs/biosar-full-${job_id}.${name}"
        local_log="$log_dir/biosar-full-${job_id}.${name}"
        if rsync -az -e 'ssh -o BatchMode=yes -o ConnectTimeout=20' \
            "$VULCAN_HOST:$remote_log" "$local_log" 2>/dev/null; then
            printf '\n===== %s =====\n' "$local_log"
            if [[ -s "$local_log" ]]; then
                tail -n "$VULCAN_LOG_TAIL_LINES" "$local_log"
            else
                echo '(log file is empty)'
            fi
        else
            printf '\n===== %s =====\n(log file not available on Vulcan)\n' "$remote_log"
        fi
    done
}

while true; do
    state=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$VULCAN_HOST" \
        "squeue -h -j $job_id -o %T")
    if [[ -z "$state" ]]; then
        break
    fi
    printf 'Job %s: %s\n' "$job_id" "$state"
    sleep "$VULCAN_POLL_SECONDS"
done

# Accounting can lag briefly after a job leaves the queue.
final_state=""
for _attempt in {1..6}; do
    final_state=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$VULCAN_HOST" \
        "sacct -n -X -j $job_id --format=State" | awk 'NF {print $1; exit}' | sed 's/+.*//')
    [[ -n "$final_state" ]] && break
    sleep 5
done

fetch_and_show_logs

if [[ -z "$final_state" ]]; then
    printf 'Job %s left the queue; final state is not available yet. Check with: ssh %s sacct -j %s\n' \
        "$job_id" "$VULCAN_HOST" "$job_id"
    exit 0
fi
printf 'Job %s finished: %s\n' "$job_id" "$final_state"
if [[ "$final_state" != COMPLETED ]]; then
    echo "Job did not complete successfully. Check jobs/biosar-full-${job_id}.out and .err on Vulcan." >&2
    exit 1
fi

#!/usr/bin/env bash
# Submit a Slurm job and wait until it leaves the queue.
# Usage: jobs/submit_monitor.sh jobs/biosarcirco.sbatch [sbatch options...]

set -euo pipefail

if (($# < 1)); then
    echo "Usage: $0 FILE.sbatch [sbatch options...]" >&2
    exit 2
fi

script=$1
shift
if [[ ! -f "$script" ]]; then
    echo "Slurm script not found: $script" >&2
    exit 2
fi
command -v sbatch >/dev/null || { echo "sbatch is unavailable; run this on Vulcan." >&2; exit 1; }
command -v squeue >/dev/null || { echo "squeue is unavailable; load/use Slurm on Vulcan." >&2; exit 1; }

submission=$(sbatch --parsable "$@" "$script")
job_id=${submission%%;*}
echo "Submitted job $job_id"

while state=$(squeue --noheader --jobs "$job_id" --format="%T") && [[ -n "$state" ]]; do
    echo "[$(date '+%F %T')] $job_id: $state"
    sleep "${POLL_SECONDS:-30}"
done

echo "Job $job_id has left the queue. Final accounting (may take a moment to appear):"
if command -v sacct >/dev/null; then
    sacct --noheader --jobs "$job_id" --format=JobID,State,ExitCode
else
    echo "sacct is unavailable; check the job with: squeue --jobs $job_id"
fi

echo "Check the stdout/stderr paths configured in $script for the run logs."

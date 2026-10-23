#!/usr/bin/env bash
set -euo pipefail
# Submit wrapper for jobs/anchor_recall.sh.
# Same option style as submit_run_inference.sh: SLURM resource flags plus a few
# script args, exported as env vars that the job script reads with fallbacks.
#
# ./jobs/submit_anchor_recall.sh
# ./jobs/submit_anchor_recall.sh -j anchor_recall_full --num-images 0 -t 03:00:00
# ./jobs/submit_anchor_recall.sh -j anchor_recall_smoke -p gpuA100x4-interactive -t 00:20:00 --num-images 20
# ./jobs/submit_anchor_recall.sh --model-types "ups_30k_ps13" --dry-run

DEFAULT_JOB_SCRIPT="$HOME/jobs/anchor_recall.sh"

# SLURM defaults (match the #SBATCH directives in anchor_recall.sh)
DEFAULT_JOB_NAME="anchor_recall"
DEFAULT_LOG_DIR="/projects/bfhm/yse2/logs/anchor_recall"
DEFAULT_ACCOUNT="bfhm-delta-gpu"
DEFAULT_PARTITION="gpuA100x4"
DEFAULT_GPUS=1
DEFAULT_CONSTRAINT="projects,work"
DEFAULT_EXCLUDE="gpua003,gpua008,gpua004"
DEFAULT_NODES=1
DEFAULT_NTASKS_PER_NODE=1
DEFAULT_CPUS_PER_TASK=16
DEFAULT_MEM="32G"
DEFAULT_TIME="01:00:00"
DEFAULT_MAIL_USER="yse2@illinois.edu"
DEFAULT_MAIL_TYPE="ALL"

# Script defaults (match the hardcoded defaults in anchor_recall.sh)
DEFAULT_MODEL_TYPES="ups_30k_ps13 ups_30k"
DEFAULT_NUM_IMAGES=500
DEFAULT_BATCH_SIZE=8
DEFAULT_NUM_WORKERS=8

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

SLURM Options:
    -s, --script PATH         Job script to submit (default: $DEFAULT_JOB_SCRIPT)
    -j, --job-name NAME       SLURM job name (default: $DEFAULT_JOB_NAME)
        --log-dir PATH        Directory for stdout/stderr logs (default: $DEFAULT_LOG_DIR)
    -A, --account NAME        SLURM account (default: $DEFAULT_ACCOUNT)
    -p, --partition NAME      SLURM partition (default: $DEFAULT_PARTITION)
    -g, --gpus N              Number of GPUs (default: $DEFAULT_GPUS)
        --constraint EXPR     Node feature constraint (default: $DEFAULT_CONSTRAINT)
        --exclude NODES       Comma-separated nodes to exclude (default: $DEFAULT_EXCLUDE)
        --nodes N             Number of nodes (default: $DEFAULT_NODES)
        --ntasks-per-node N   Tasks per node (default: $DEFAULT_NTASKS_PER_NODE)
    -c, --cpus N              CPUs per task (default: $DEFAULT_CPUS_PER_TASK)
    -m, --mem SIZE            Memory allocation (default: $DEFAULT_MEM)
    -t, --time HH:MM:SS       Time limit (default: $DEFAULT_TIME)
        --mail-user EMAIL     Email for job notifications (default: $DEFAULT_MAIL_USER)
        --mail-type TYPE      Notification events (default: $DEFAULT_MAIL_TYPE)

Script Options:
    --model-types "A B"       Space-separated model types, PS13 first
                              (default: "$DEFAULT_MODEL_TYPES")
    --num-images N            Validation images to measure; 0 means all
                              (default: $DEFAULT_NUM_IMAGES)
    --batch-size N            Test batch size (default: $DEFAULT_BATCH_SIZE)
    --num-workers N           DataLoader workers (default: $DEFAULT_NUM_WORKERS)
    --data-fn PATH            Annotation JSON to slice (default: the val_4k set)

Other:
    --dry-run                 Print the sbatch command without submitting
    -h, --help                Show this help
EOF
}

JOB_SCRIPT="$DEFAULT_JOB_SCRIPT"
JOB_NAME="$DEFAULT_JOB_NAME"
LOG_DIR="$DEFAULT_LOG_DIR"
ACCOUNT="$DEFAULT_ACCOUNT"
PARTITION="$DEFAULT_PARTITION"
GPUS="$DEFAULT_GPUS"
CONSTRAINT="$DEFAULT_CONSTRAINT"
EXCLUDE="$DEFAULT_EXCLUDE"
NODES="$DEFAULT_NODES"
NTASKS_PER_NODE="$DEFAULT_NTASKS_PER_NODE"
CPUS="$DEFAULT_CPUS_PER_TASK"
MEMORY="$DEFAULT_MEM"
TIME_LIMIT="$DEFAULT_TIME"
MAIL_USER="$DEFAULT_MAIL_USER"
MAIL_TYPE="$DEFAULT_MAIL_TYPE"
MODEL_TYPES="$DEFAULT_MODEL_TYPES"
NUM_IMAGES="$DEFAULT_NUM_IMAGES"
BATCH_SIZE="$DEFAULT_BATCH_SIZE"
NUM_WORKERS="$DEFAULT_NUM_WORKERS"
DATA_FN=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--script)          JOB_SCRIPT="$2"; shift 2 ;;
        -j|--job-name)        JOB_NAME="$2"; shift 2 ;;
        --log-dir)            LOG_DIR="$2"; shift 2 ;;
        -A|--account)         ACCOUNT="$2"; shift 2 ;;
        -p|--partition)       PARTITION="$2"; shift 2 ;;
        -g|--gpus)            GPUS="$2"; shift 2 ;;
        --constraint)         CONSTRAINT="$2"; shift 2 ;;
        --exclude)            EXCLUDE="$2"; shift 2 ;;
        --nodes)              NODES="$2"; shift 2 ;;
        --ntasks-per-node)    NTASKS_PER_NODE="$2"; shift 2 ;;
        -c|--cpus)            CPUS="$2"; shift 2 ;;
        -m|--mem)             MEMORY="$2"; shift 2 ;;
        -t|--time)            TIME_LIMIT="$2"; shift 2 ;;
        --mail-user)          MAIL_USER="$2"; shift 2 ;;
        --mail-type)          MAIL_TYPE="$2"; shift 2 ;;
        --model-types)        MODEL_TYPES="$2"; shift 2 ;;
        --num-images)         NUM_IMAGES="$2"; shift 2 ;;
        --batch-size)         BATCH_SIZE="$2"; shift 2 ;;
        --num-workers)        NUM_WORKERS="$2"; shift 2 ;;
        --data-fn)            DATA_FN="$2"; shift 2 ;;
        --dry-run)            DRY_RUN=true; shift ;;
        -h|--help)            show_usage; exit 0 ;;
        *) echo "Unknown option: $1"; show_usage; exit 1 ;;
    esac
done

# Validation
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "Error: Job script not found: $JOB_SCRIPT"
    exit 1
fi
if [[ ! "$GPUS" =~ ^[0-9]+$ ]] || [[ "$GPUS" -lt 1 ]]; then
    echo "Error: --gpus must be a positive integer"
    exit 1
fi
for MT in $MODEL_TYPES; do
    if [[ ! "$MT" =~ ^(ups_30k_ps13|ups_30k)$ ]]; then
        echo "Error: --model-types entries must be one of: ups_30k_ps13, ups_30k"
        exit 1
    fi
done

export MODEL_TYPES_STR="$MODEL_TYPES"
export NUM_IMAGES
export BATCH_SIZE
export NUM_WORKERS
export DATA_FN

mkdir -p "$LOG_DIR"

SBATCH_CMD=(
    sbatch
    "--job-name=$JOB_NAME"
    "--output=${LOG_DIR}/${JOB_NAME}.%j.%N.out"
    "--error=${LOG_DIR}/${JOB_NAME}.%j.%N.err"
    "--account=$ACCOUNT"
    "--partition=$PARTITION"
    "--gres=gpu:$GPUS"
    "--constraint=$CONSTRAINT"
    "--exclude=$EXCLUDE"
    "--nodes=$NODES"
    "--ntasks-per-node=$NTASKS_PER_NODE"
    "--cpus-per-task=$CPUS"
    "--mem=$MEMORY"
    "--time=$TIME_LIMIT"
    "--mail-user=$MAIL_USER"
    "--mail-type=$MAIL_TYPE"
    "$JOB_SCRIPT"
)

echo "Anchor/RPN recall configuration:"
echo "  Model types  : $MODEL_TYPES"
echo "  Num images   : $NUM_IMAGES"
echo "  Batch size   : $BATCH_SIZE"
echo "  Num workers  : $NUM_WORKERS"
[[ -n "$DATA_FN" ]] && echo "  Data file    : $DATA_FN"
echo ""
echo "SLURM configuration:"
echo "  Job name     : $JOB_NAME"
echo "  Log dir      : $LOG_DIR"
echo "  Account      : $ACCOUNT"
echo "  Partition    : $PARTITION"
echo "  GPUs         : $GPUS"
echo "  CPUs         : $CPUS"
echo "  Memory       : $MEMORY"
echo "  Time limit   : $TIME_LIMIT"
echo ""
echo "Command: ${SBATCH_CMD[*]}"

if [[ "$DRY_RUN" == true ]]; then
    echo "(dry run — not submitted)"
    exit 0
fi

"${SBATCH_CMD[@]}"

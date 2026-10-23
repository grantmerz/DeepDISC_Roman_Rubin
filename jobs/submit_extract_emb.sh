#!/usr/bin/env bash
set -euo pipefail
# Submit wrapper for jobs/extract_emb.sh.
# Accepts all SLURM resource options and extract_emb.py args
# Extract emb args are exported as environment variables consumed by the job script

# Minimal and correct examples for submit_extract_emb.sh:
# ./jobs/submit_extract_emb.sh -j test_extract_30k -g 2 --dry-run
# ./jobs/submit_extract_emb.sh -j test_extract_30k_flatten -g 2 --run-name clip5_flatten_30k_4h200_bs64_ep15
# ./jobs/submit_extract_emb.sh -j test_extract_30k_flat50ep -g 2 --run-name clip5_flatten_30k_4h200_bs64_ep15_resume
# ./jobs/submit_extract_emb.sh -j test_extract_30k_og -g 2 --run-name clip5_30k_4h200_bs64_ep50
# ./jobs/submit_extract_emb.sh -j test_extract_multi -g 2 --run-names clip5_30k_4h200_bs192_ep15,clip5_flatten_30k_4h200_bs64_ep15,clip5_flatten_30k_4h200_bs64_ep15_resume,clip5_30k_4h200_bs64_ep50 -t 00:30:00 --save-repr pooled_q -m 96G
# ./jobs/submit_extract_emb.sh -j test_extract_multi -g 2 --run-names clip5_30k_4h200_bs192_ep15,clip5_flatten_30k_4h200_bs64_ep15,clip5_flatten_30k_4h200_bs64_ep15_resume,clip5_30k_4h200_bs64_ep50 -t 00:45:00 --save-repr pooled_q -m 96G
# ./jobs/submit_extract_emb.sh -j txtract_lprj -g 2 --run-names clip5_30k_4h200_bs192_ep15_lprj
# ./jobs/submit_extract_emb.sh -j txtract_lprj -g 2 --run-names clip5_30k_4h200_bs192_ep15_lprj -t 00:45:00 --save-repr pooled_q -m 96G
# ./jobs/submit_extract_emb.sh -j txtract_ep15 -g 2 --run-names lsst5_30k_4h200_bs192_ep15 -t 00:45:00 --save-repr pooled_q -m 96G
DEFAULT_JOB_SCRIPT="$HOME/jobs/extract_emb.sh"

# SLURM defaults (match the #SBATCH defaults in extract_emb.sh)
DEFAULT_JOB_NAME="test_extract_30k"
DEFAULT_LOG_DIR="/projects/bfhm/yse2/logs/extract_emb/test/"
DEFAULT_ACCOUNT="bdsp-delta-gpu"
DEFAULT_PARTITION="gpuA100x4"
DEFAULT_GPUS=4
DEFAULT_CONSTRAINT="projects,work"
DEFAULT_EXCLUDE="gpua003"
DEFAULT_NODES=1
DEFAULT_NTASKS_PER_NODE=1
DEFAULT_CPUS_PER_TASK=32
DEFAULT_MEM="64G"
DEFAULT_TIME="00:15:00"
DEFAULT_MAIL_USER="yse2@illinois.edu"
DEFAULT_MAIL_TYPE="ALL"

# Extract emb defaults (match the hardcoded defaults in extract_emb.sh)
DEFAULT_RUN_NAME="clip5_30k_4h200_bs192_ep15"
DEFAULT_DATA_SPLIT="test"
DEFAULT_SAVE_REPR="z_q"

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

SLURM Options:
    -s, --script PATH              Job script to submit (default: $DEFAULT_JOB_SCRIPT)
    -j, --job-name NAME            SLURM job name (default: $DEFAULT_JOB_NAME)
        --log-dir PATH             Directory for stdout/stderr logs (default: $DEFAULT_LOG_DIR)
    -A, --account NAME             SLURM account ($DEFAULT_ACCOUNT)
    -p, --partition NAME           SLURM partition (default: $DEFAULT_PARTITION)
    -g, --gpus N                   Number of GPUs — sets both --gres and --num_gpus (default: $DEFAULT_GPUS)
        --constraint EXPR          Node feature constraint, e.g. "projects,work" (default: $DEFAULT_CONSTRAINT)
        --exclude NODES            Comma-separated nodes to exclude, e.g. gpua003 (optional, default: $DEFAULT_EXCLUDE)
        --nodes N                  Number of nodes (default: $DEFAULT_NODES)
        --ntasks-per-node N        Tasks per node (default: $DEFAULT_NTASKS_PER_NODE)
    -c, --cpus N                   CPUs per task (default: $DEFAULT_CPUS_PER_TASK)
    -m, --mem SIZE                 Memory allocation (default: $DEFAULT_MEM)
    -t, --time HH:MM:SS            Time limit (default: $DEFAULT_TIME)
        --mail-user EMAIL          Email for job notifications (default: $DEFAULT_MAIL_USER)
        --mail-type TYPE           Notification events: ALL, BEGIN, END, FAIL (default: $DEFAULT_MAIL_TYPE if --mail-user set)

Extract Emb Options:
    --run-name NAME                Run name (default: $DEFAULT_RUN_NAME)
    --run-names CSV                Comma-separated run names to execute in one job (overrides --run-name)
    --data-split SPLIT             Data split: eval or test (default: $DEFAULT_DATA_SPLIT)
    --save-repr MODE               Representation mode: both, z_q, or pooled_q (default: $DEFAULT_SAVE_REPR)
    --cfgfile PATH                 Override config file passed to extract_emb.py (optional)
    --test-data-fn PATH            Override test data file passed to extract_emb.py (optional)

Misc:
    --dry-run                      Print configuration and sbatch command without submitting
    -h, --help                     Show this message

Examples:
  $0
    $0 -p gpuA100x4 -g 2 -t 02:00:00 --run-name my_run --data-split test --dry-run
EOF
}

# Defaults
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

RUN_NAME="$DEFAULT_RUN_NAME"
RUN_NAMES=""
DATA_SPLIT="$DEFAULT_DATA_SPLIT"
SAVE_REPR="$DEFAULT_SAVE_REPR"
CFGFILE=""
TEST_DATA_FN=""
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--script)            JOB_SCRIPT="$2";           shift 2 ;;
        -j|--job-name)          JOB_NAME="$2";             shift 2 ;;
        --log-dir)              LOG_DIR="$2";              shift 2 ;;
        -A|--account)           ACCOUNT="$2";              shift 2 ;;
        -p|--partition)         PARTITION="$2";            shift 2 ;;
        -g|--gpus)              GPUS="$2";                 shift 2 ;;
        --constraint)           CONSTRAINT="$2";           shift 2 ;;
        --exclude)              EXCLUDE="$2";              shift 2 ;;
        --nodes)                NODES="$2";                shift 2 ;;
        --ntasks-per-node)      NTASKS_PER_NODE="$2";      shift 2 ;;
        -c|--cpus)              CPUS="$2";                 shift 2 ;;
        -m|--mem)               MEMORY="$2";               shift 2 ;;
        -t|--time)              TIME_LIMIT="$2";           shift 2 ;;
        --mail-user)            MAIL_USER="$2";            shift 2 ;;
        --mail-type)            MAIL_TYPE="$2";            shift 2 ;;
        --run-name)             RUN_NAME="$2";             shift 2 ;;
        --run-names)            RUN_NAMES="$2";            shift 2 ;;
        --data-split)           DATA_SPLIT="$2";           shift 2 ;;
        --save-repr)            SAVE_REPR="$2";            shift 2 ;;
        --cfgfile)              CFGFILE="$2";              shift 2 ;;
        --test-data-fn)         TEST_DATA_FN="$2";         shift 2 ;;
        --dry-run)              DRY_RUN="true";            shift ;;
        -h|--help)              show_usage; exit 0 ;;
        *)
            echo "Error: Unknown option '$1'"
            show_usage
            exit 1
            ;;
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
if [[ ! "$DATA_SPLIT" =~ ^(eval|test)$ ]]; then
    echo "Error: --data-split must be 'eval' or 'test'"
    exit 1
fi
if [[ ! "$SAVE_REPR" =~ ^(both|z_q|pooled_q)$ ]]; then
    echo "Error: --save-repr must be one of: both, z_q, pooled_q"
    exit 1
fi
if [[ -n "$RUN_NAMES" ]]; then
    RUN_NAMES="$RUN_NAMES"
else
    RUN_NAMES="$RUN_NAME"
fi

# Export extract_emb args as env vars; extract_emb.sh reads these with fallback defaults.
export RUN_NAME
export RUN_NAMES
export DATA_SPLIT
export SAVE_REPR
export CFG_FILE_OVERRIDE="$CFGFILE"
export TEST_DATA_FN_OVERRIDE="$TEST_DATA_FN"
export NUM_GPUS="$GPUS"

# Build sbatch command. CLI options override the #SBATCH directives in the job script
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
)
SBATCH_CMD+=("$JOB_SCRIPT")

echo "Extract Emb configuration:"
[[ -n "$RUN_NAMES" ]] && echo "  Run names       : $RUN_NAMES"
echo "  Data split      : $DATA_SPLIT"
echo "  Save repr       : $SAVE_REPR"
[[ -n "$CFGFILE" ]] && echo "  Cfg file        : $CFGFILE"
[[ -n "$TEST_DATA_FN" ]] && echo "  Test data fn    : $TEST_DATA_FN"
echo "  Num GPUs        : $GPUS"
echo ""
echo "SLURM configuration:"
echo "  Job name        : $JOB_NAME"
echo "  Log dir         : $LOG_DIR"
echo "  Account         : $ACCOUNT"
echo "  Partition       : $PARTITION"
echo "  GPUs            : $GPUS"
echo "  Constraint      : $CONSTRAINT"
echo "  Exclude nodes   : $EXCLUDE"
echo "  Nodes           : $NODES"
echo "  Tasks/node      : $NTASKS_PER_NODE"
echo "  CPUs per task   : $CPUS"
echo "  Memory          : $MEMORY"
echo "  Time            : $TIME_LIMIT"
echo "  Mail user       : $MAIL_USER  (type: $MAIL_TYPE)"
echo ""
echo "Command:$(printf ' %q' "${SBATCH_CMD[@]}")"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run — no job submitted."
    exit 0
fi

"${SBATCH_CMD[@]}"

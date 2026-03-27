#!/usr/bin/env bash
set -euo pipefail
# Submit wrapper for jobs/run_inference.sh.
# Accepts all SLURM resource options and run_inference.py args
# Inference args are exported as environment variables consumed by the job script
# submit_run_inference.sh -j test_infer_sall --log-dir /projects/bfhm/yse2/logs/inference/test -g 2 -t 00:40:00 --model-type standard_all --data-split test --score-thresholds "0.45 0.55" --nms-thresholds "0.65 0.65" --no-combo --dry-run
# ./jobs/submit_run_inference.sh -j eval_infer_c30k_lgroi -g 2 -p gpuA100x4-interactive -t 00:45:00 --model-type clip_30k --run-name clip5_flatten_30k_4h200_bs64_ep15_resume --score-thresholds "0.45 0.5 0.55 0.6 0.65" --nms-thresholds "0.45 0.5 0.55 0.6 0.65"
# ./jobs/submit_run_inference.sh -j test_infer_c30k_lgroi --log-dir /projects/bfhm/yse2/logs/inference/test -g 2 -p gpuA100x4-interactive -t 00:25:00 --model-type clip_30k --data-split test --run-name clip5_flatten_30k_4h200_bs64_ep15_resume --score-thresholds "0.4" --nms-thresholds "0.55"
DEFAULT_JOB_SCRIPT="$HOME/jobs/run_inference.sh"

# SLURM defaults (match the #SBATCH defaults in run_inference.sh)
DEFAULT_JOB_NAME="eval_infer_s30k"
DEFAULT_LOG_DIR="/projects/bfhm/yse2/logs/inference/eval"
DEFAULT_ACCOUNT="bdsp-delta-gpu"
DEFAULT_PARTITION="gpuA100x4"
DEFAULT_GPUS=4
DEFAULT_CONSTRAINT="projects,work"
DEFAULT_EXCLUDE="gpua003"
DEFAULT_NODES=1
DEFAULT_NTASKS_PER_NODE=1
DEFAULT_CPUS_PER_TASK=32
DEFAULT_MEM="64G"
DEFAULT_TIME="04:30:00"
DEFAULT_MAIL_USER="yse2@illinois.edu"
DEFAULT_MAIL_TYPE="ALL"

# Inference defaults (match the hardcoded defaults in run_inference.sh)
DEFAULT_MODEL_TYPE="standard_30k"
DEFAULT_DATA_SPLIT="eval"
DEFAULT_SCORE_THRESHOLDS="0.55"
DEFAULT_NMS_THRESHOLDS="0.65"
DEFAULT_TOPK=2000
DEFAULT_DATA_ROOT_DIR="~/lsst_data"
DEFAULT_ANNS_FOLDER="annotations_lvl5"

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

Inference Options:
    --model-type TYPE              Model type: standard_30k, standard_all, clip_30k, clip_all (default: $DEFAULT_MODEL_TYPE)
    --run-name NAME                Override run name (used instead of model-type default)
    --data-split SPLIT             Data split: eval or test (default: $DEFAULT_DATA_SPLIT)
    --cfgfile PATH                 Override config file passed to run_inference.py (optional)
    --score-thresholds "S1 S2..."  Space-separated score thresholds (default: "$DEFAULT_SCORE_THRESHOLDS")
    --nms-thresholds "N1 N2..."    Space-separated NMS thresholds (default: "$DEFAULT_NMS_THRESHOLDS")
    --no-combo                     Pair score/NMS thresholds by index instead of cartesian product
    --topk-per-img N               Top detections per image (default: $DEFAULT_TOPK)
    --data-root-dir PATH           Root data directory (default: $DEFAULT_DATA_ROOT_DIR)
    --anns-folder NAME             Annotations subfolder (default: $DEFAULT_ANNS_FOLDER)
    --resume                       Skip threshold combos whose output .json already exists

Misc:
    --dry-run                      Print configuration and sbatch command without submitting
    -h, --help                     Show this message

Examples:
  $0
  $0 -p gpuA100x4 -g 2 -t 02:00:00 --model-type standard_30k --data-split eval
  $0 --score-thresholds "0.4 0.5 0.6" --nms-thresholds "0.3 0.4 0.5" --no-combo --resume
  $0 --dry-run
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
MODEL_TYPE="$DEFAULT_MODEL_TYPE"
RUN_NAME=""
DATA_SPLIT="$DEFAULT_DATA_SPLIT"
CFGFILE=""
SCORE_THRESHOLDS="$DEFAULT_SCORE_THRESHOLDS"
NMS_THRESHOLDS="$DEFAULT_NMS_THRESHOLDS"
NO_COMBO="false"
TOPK="$DEFAULT_TOPK"
DATA_ROOT_DIR="$DEFAULT_DATA_ROOT_DIR"
ANNS_FOLDER="$DEFAULT_ANNS_FOLDER"
RESUME="false"
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
        --model-type)           MODEL_TYPE="$2";           shift 2 ;;
        --run-name)             RUN_NAME="$2";             shift 2 ;;
        --data-split)           DATA_SPLIT="$2";           shift 2 ;;
        --cfgfile)              CFGFILE="$2";              shift 2 ;;
        --score-thresholds)     SCORE_THRESHOLDS="$2";     shift 2 ;;
        --nms-thresholds)       NMS_THRESHOLDS="$2";       shift 2 ;;
        --no-combo)             NO_COMBO="true";           shift ;;
        --topk-per-img)         TOPK="$2";                 shift 2 ;;
        --data-root-dir)        DATA_ROOT_DIR="$2";        shift 2 ;;
        --anns-folder)          ANNS_FOLDER="$2";          shift 2 ;;
        --resume)               RESUME="true";             shift ;;
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
if [[ ! "$MODEL_TYPE" =~ ^(standard_30k|standard_all|clip_30k|clip_all)$ ]]; then
    echo "Error: --model-type must be one of: standard_30k, standard_all, clip_30k, clip_all"
    exit 1
fi

# Export inference args as env vars; run_inference.sh reads these with fallback defaults.
# Thresholds are exported as space-separated strings and split into arrays by the job script.
export MODEL_TYPE
export RUN_NAME_OVERRIDE="$RUN_NAME"
export DATA_SPLIT
export CFGFILE
export SCORE_THRESHOLDS_STR="$SCORE_THRESHOLDS"
export NMS_THRESHOLDS_STR="$NMS_THRESHOLDS"
export NO_COMBO
export TOPK_PER_IMG="$TOPK"
export DATA_ROOT_DIR
export ANNS_FOLDER
export NUM_GPUS="$GPUS"
export RESUME

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

echo "Inference configuration:"
echo "  Model type      : $MODEL_TYPE"
[[ -n "$RUN_NAME" ]] && echo "  Run name        : $RUN_NAME"
echo "  Data split      : $DATA_SPLIT"
[[ -n "$CFGFILE" ]] && echo "  Cfg file        : $CFGFILE"
echo "  Score thresholds: $SCORE_THRESHOLDS"
echo "  NMS thresholds  : $NMS_THRESHOLDS"
echo "  No combo        : $NO_COMBO"
echo "  Top-K per image : $TOPK"
echo "  Data root dir   : $DATA_ROOT_DIR"
echo "  Anns folder     : $ANNS_FOLDER"
echo "  Resume          : $RESUME"
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
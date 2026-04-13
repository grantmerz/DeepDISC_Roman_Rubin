#!/usr/bin/env bash
set -euo pipefail

# Submit wrapper for jobs/add_dd_mags.sh.
# Supports one or more runs via --run-names and submits one SLURM job per run.
# Example:
#   submit_add_dd_mags.sh --run-names "lsst5_30k_4h200_bs192_ep50 clip5_30k_4h200_bs64_ep50" \
#       --score-thresholds "0.4 0.5" --nms-thresholds "0.55 0.6" --no-combo --dry-run
#   submit_add_dd_mags.sh -j sall_test_mags --run-names "lsst5_all_4h200_bs192_ep20" --preds-dir test --filenames-csv /u/yse2/test_fns.csv --score-thresholds "0.45 0.55" --nms-thresholds "0.65 0.65" --no-combo --dry-run
#   submit_add_dd_mags.sh -j c30k_test_mags --run-names "clip5_flatten_30k_4h200_bs64_ep15_resume" --preds-dir test --filenames-csv /u/yse2/test8k_fns.csv --score-thresholds "0.4" --nms-thresholds "0.55" --dry-run
#   submit_add_dd_mags.sh -j c30k_test_mags --run-names "clip5_flatten_30k_4h200_bs64_ep15_resume clip5_30k_4h200_bs64_ep50" --preds-dir test --filenames-csv /u/yse2/test8k_fns.csv --score-thresholds "0.4" --nms-thresholds "0.55" --dry-run

# submit_add_dd_mags.sh -j comb30k_eval_mags --run-names "comb_30k_4h200_bs144_ep50" --preds-dir eval --filenames-csv /u/yse2/val4k_fns_wcs.csv --score-thresholds "0.45 0.5 0.55 0.6 0.65" --nms-thresholds "0.45 0.5 0.55 0.6 0.65" --upsample --dry-run
# submit_add_dd_mags.sh -j comb30k_test_mags --run-names "comb_30k_4h200_bs144_ep50" --preds-dir test --filenames-csv /u/yse2/test8k_fns_wcs.csv -t 02:30:00 --score-thresholds "0.4 0.55 0.65" --nms-thresholds "0.55 0.65 0.65" --upsample --no-combo --dry-run

# submit_add_dd_mags.sh -j d30k_eval_mags --run-names "distill_30k_4h200_bs192_ep50" --preds-dir eval --filenames-csv /u/yse2/val4k_fns.csv --score-thresholds "0.4" --nms-thresholds "0.55" --dry-run
# submit_add_dd_mags.sh -j d30k_test_mags --run-names "distill_30k_4h200_bs192_ep50" --preds-dir test --filenames-csv /u/yse2/test8k_fns.csv --score-thresholds "0.65" --nms-thresholds "0.65" --dry-run
# submit_add_dd_mags.sh -j d30k_test_mags --run-names "lsst5_30k_4h200_bs192_ep50" --preds-dir test --filenames-csv /u/yse2/test8k_fns.csv --score-thresholds "0.65" --nms-thresholds "0.65" --dry-run
# submit_add_dd_mags.sh -j c30k_test_mags --run-names "clip5_30k_4h200_bs192_ep15_lprj" --preds-dir test --filenames-csv /u/yse2/test8k_fns.csv --score-thresholds "0.4" --nms-thresholds "0.55"
# submit_add_dd_mags.sh -j c30k_test_mags --run-names "lsst5_30k_4h200_bs192_ep50" --preds-dir test --filenames-csv /u/yse2/test8k_fns.csv --score-thresholds "0.4" --nms-thresholds "0.55"

# SLURM defaults (match jobs/add_dd_mags.sh)
DEFAULT_JOB_SCRIPT="$HOME/jobs/add_dd_mags.sh"
DEFAULT_JOB_NAME="s30k_eval_mags"
DEFAULT_LOG_DIR="/projects/bfhm/yse2/logs/add_mags"
DEFAULT_ACCOUNT="bfhm-delta-cpu"
DEFAULT_PARTITION="cpu"
DEFAULT_CONSTRAINT="work"
DEFAULT_NODES=1
DEFAULT_NTASKS_PER_NODE=1
DEFAULT_CPUS_PER_TASK=32
DEFAULT_MEM="64G"
DEFAULT_TIME="00:45:00"
DEFAULT_MAIL_USER="yse2@illinois.edu"
DEFAULT_MAIL_TYPE="ALL"

# add_dd_mags.py defaults
DEFAULT_ROOT_RUN_DIR="~/lsst_runs/"
DEFAULT_PREDS_DIR="eval"
DEFAULT_FILENAMES_CSV="~/val4k_fns.csv"
DEFAULT_IO_THREADS=16
DEFAULT_N_PROCESSES=16
DEFAULT_UPSAMPLE="false"
DEFAULT_SCORE_THRESHOLDS="0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"
DEFAULT_NMS_THRESHOLDS="0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75"

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Required:
    --run-names "R1 R2 ..."        Space-separated run names; one sbatch job per run

SLURM Options:
    -s, --script PATH              Job script to submit (default: $DEFAULT_JOB_SCRIPT)
    -j, --job-name NAME            Base SLURM job name (default: $DEFAULT_JOB_NAME)
        --log-dir PATH             Directory for stdout/stderr logs (default: $DEFAULT_LOG_DIR)
    -A, --account NAME             SLURM account (default: $DEFAULT_ACCOUNT)
    -p, --partition NAME           SLURM partition (default: $DEFAULT_PARTITION)
        --constraint EXPR          Node feature constraint (default: $DEFAULT_CONSTRAINT)
        --nodes N                  Number of nodes (default: $DEFAULT_NODES)
        --ntasks-per-node N        Tasks per node (default: $DEFAULT_NTASKS_PER_NODE)
    -c, --cpus N                   CPUs per task (default: $DEFAULT_CPUS_PER_TASK)
    -m, --mem SIZE                 Memory allocation (default: $DEFAULT_MEM)
    -t, --time HH:MM:SS            Time limit (default: $DEFAULT_TIME)
        --mail-user EMAIL          Email for job notifications (default: $DEFAULT_MAIL_USER)
        --mail-type TYPE           Notification events (default: $DEFAULT_MAIL_TYPE)

add_dd_mags.py Options:
    --root-run-dir PATH            Root run directory (default: $DEFAULT_ROOT_RUN_DIR)
    --preds-dir NAME               Subdirectory under preds/ (default: $DEFAULT_PREDS_DIR)
    --filenames-csv PATH           CSV with file_name column (default: $DEFAULT_FILENAMES_CSV)
    --score-thresholds "S1 S2"     Space-separated score thresholds (default: "$DEFAULT_SCORE_THRESHOLDS")
    --nms-thresholds "N1 N2"       Space-separated NMS thresholds (default: "$DEFAULT_NMS_THRESHOLDS")
    --no-combo                     Pair score/NMS by index instead of cartesian product
    --io-threads N                 Threads for image loading (default: $DEFAULT_IO_THREADS)
    --n-processes N                Worker processes (default: $DEFAULT_N_PROCESSES)
    --upsample                     Enable LSST upsampling before magnitude computation (default: disabled)
    --resume                       Skip completed output combos

Misc:
    --dry-run                      Print sbatch commands without submitting
    -h, --help                     Show this message
EOF
}

# Defaults
JOB_SCRIPT="$DEFAULT_JOB_SCRIPT"
JOB_NAME="$DEFAULT_JOB_NAME"
LOG_DIR="$DEFAULT_LOG_DIR"
ACCOUNT="$DEFAULT_ACCOUNT"
PARTITION="$DEFAULT_PARTITION"
CONSTRAINT="$DEFAULT_CONSTRAINT"
NODES="$DEFAULT_NODES"
NTASKS_PER_NODE="$DEFAULT_NTASKS_PER_NODE"
CPUS="$DEFAULT_CPUS_PER_TASK"
MEMORY="$DEFAULT_MEM"
TIME_LIMIT="$DEFAULT_TIME"
MAIL_USER="$DEFAULT_MAIL_USER"
MAIL_TYPE="$DEFAULT_MAIL_TYPE"

ROOT_RUN_DIR="$DEFAULT_ROOT_RUN_DIR"
PREDS_DIR="$DEFAULT_PREDS_DIR"
FILENAMES_CSV="$DEFAULT_FILENAMES_CSV"
SCORE_THRESHOLDS="$DEFAULT_SCORE_THRESHOLDS"
NMS_THRESHOLDS="$DEFAULT_NMS_THRESHOLDS"
IO_THREADS="$DEFAULT_IO_THREADS"
N_PROCESSES="$DEFAULT_N_PROCESSES"
UPSAMPLE="$DEFAULT_UPSAMPLE"
NO_COMBO="false"
RESUME="false"
DRY_RUN="false"
RUN_NAMES_RAW=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--script)            JOB_SCRIPT="$2";           shift 2 ;;
        -j|--job-name)          JOB_NAME="$2";             shift 2 ;;
        --log-dir)              LOG_DIR="$2";              shift 2 ;;
        -A|--account)           ACCOUNT="$2";              shift 2 ;;
        -p|--partition)         PARTITION="$2";            shift 2 ;;
        --constraint)           CONSTRAINT="$2";           shift 2 ;;
        --nodes)                NODES="$2";                shift 2 ;;
        --ntasks-per-node)      NTASKS_PER_NODE="$2";      shift 2 ;;
        -c|--cpus)              CPUS="$2";                 shift 2 ;;
        -m|--mem)               MEMORY="$2";               shift 2 ;;
        -t|--time)              TIME_LIMIT="$2";           shift 2 ;;
        --mail-user)            MAIL_USER="$2";            shift 2 ;;
        --mail-type)            MAIL_TYPE="$2";            shift 2 ;;

        --run-names)            RUN_NAMES_RAW="$2";        shift 2 ;;
        --root-run-dir)         ROOT_RUN_DIR="$2";         shift 2 ;;
        --preds-dir)            PREDS_DIR="$2";            shift 2 ;;
        --filenames-csv)        FILENAMES_CSV="$2";        shift 2 ;;
        --score-thresholds)     SCORE_THRESHOLDS="$2";     shift 2 ;;
        --nms-thresholds)       NMS_THRESHOLDS="$2";       shift 2 ;;
        --no-combo)             NO_COMBO="true";           shift ;;
        --io-threads)           IO_THREADS="$2";           shift 2 ;;
        --n-processes)          N_PROCESSES="$2";          shift 2 ;;
        --upsample)             UPSAMPLE="true";           shift ;;
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
if [[ -z "$RUN_NAMES_RAW" ]]; then
    echo "Error: --run-names is required"
    show_usage
    exit 1
fi
if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "Error: Job script not found: $JOB_SCRIPT"
    exit 1
fi
if [[ ! "$NODES" =~ ^[0-9]+$ ]] || [[ "$NODES" -lt 1 ]]; then
    echo "Error: --nodes must be a positive integer"
    exit 1
fi
if [[ ! "$NTASKS_PER_NODE" =~ ^[0-9]+$ ]] || [[ "$NTASKS_PER_NODE" -lt 1 ]]; then
    echo "Error: --ntasks-per-node must be a positive integer"
    exit 1
fi
if [[ ! "$CPUS" =~ ^[0-9]+$ ]] || [[ "$CPUS" -lt 1 ]]; then
    echo "Error: --cpus must be a positive integer"
    exit 1
fi
if [[ ! "$IO_THREADS" =~ ^[0-9]+$ ]] || [[ "$IO_THREADS" -lt 1 ]]; then
    echo "Error: --io-threads must be a positive integer"
    exit 1
fi
if [[ ! "$N_PROCESSES" =~ ^[0-9]+$ ]] || [[ "$N_PROCESSES" -lt 1 ]]; then
    echo "Error: --n-processes must be a positive integer"
    exit 1
fi
read -ra RUN_NAMES <<< "$RUN_NAMES_RAW"
if [[ "${#RUN_NAMES[@]}" -eq 0 ]]; then
    echo "Error: --run-names did not contain any run names"
    exit 1
fi

read -ra SCORE_THRESH_ARR <<< "$SCORE_THRESHOLDS"
read -ra NMS_THRESH_ARR <<< "$NMS_THRESHOLDS"
if [[ "${#SCORE_THRESH_ARR[@]}" -eq 0 ]] || [[ "${#NMS_THRESH_ARR[@]}" -eq 0 ]]; then
    echo "Error: threshold lists cannot be empty"
    exit 1
fi
if [[ "$NO_COMBO" == "true" ]] && [[ "${#SCORE_THRESH_ARR[@]}" -ne "${#NMS_THRESH_ARR[@]}" ]]; then
    echo "Error: --no-combo requires equal list lengths (${#SCORE_THRESH_ARR[@]} scores vs ${#NMS_THRESH_ARR[@]} nms)"
    exit 1
fi

echo "Magnitude configuration:"
echo "  Run names        : ${RUN_NAMES[*]}"
echo "  Root run dir     : $ROOT_RUN_DIR"
echo "  Preds dir        : $PREDS_DIR"
echo "  Filenames CSV    : $FILENAMES_CSV"
echo "  Score thresholds : $SCORE_THRESHOLDS"
echo "  NMS thresholds   : $NMS_THRESHOLDS"
echo "  No combo         : $NO_COMBO"
echo "  IO threads       : $IO_THREADS"
echo "  Worker processes : $N_PROCESSES"
echo "  Upsample         : $UPSAMPLE"
echo "  Resume           : $RESUME"
echo ""
echo "SLURM configuration:"
echo "  Base job name    : $JOB_NAME"
echo "  Log dir          : $LOG_DIR"
echo "  Account          : $ACCOUNT"
echo "  Partition        : $PARTITION"
echo "  Constraint       : $CONSTRAINT"
echo "  Nodes            : $NODES"
echo "  Tasks/node       : $NTASKS_PER_NODE"
echo "  CPUs per task    : $CPUS"
echo "  Memory           : $MEMORY"
echo "  Time             : $TIME_LIMIT"
echo "  Mail user        : $MAIL_USER  (type: $MAIL_TYPE)"
echo ""

for run_name in "${RUN_NAMES[@]}"; do
    job_name_run="${JOB_NAME}_${run_name}"

    export RUN_NAME="$run_name"
    export ROOT_RUN_DIR
    export PREDS_DIR
    export FILENAMES_CSV
    export SCORE_THRESHOLDS_STR="$SCORE_THRESHOLDS"
    export NMS_THRESHOLDS_STR="$NMS_THRESHOLDS"
    export NO_COMBO
    export IO_THREADS
    export N_PROCESSES
    export UPSAMPLE
    export RESUME

    SBATCH_CMD=(
        sbatch
        "--job-name=$job_name_run"
        "--output=${LOG_DIR}/${job_name_run}.%j.%N.out"
        "--error=${LOG_DIR}/${job_name_run}.%j.%N.err"
        "--account=$ACCOUNT"
        "--partition=$PARTITION"
        "--constraint=$CONSTRAINT"
        "--nodes=$NODES"
        "--ntasks-per-node=$NTASKS_PER_NODE"
        "--cpus-per-task=$CPUS"
        "--mem=$MEMORY"
        "--time=$TIME_LIMIT"
        "--mail-user=$MAIL_USER"
        "--mail-type=$MAIL_TYPE"
    )
    SBATCH_CMD+=("$JOB_SCRIPT")

    echo "Run: $run_name"
    echo "Command:$(printf ' %q' "${SBATCH_CMD[@]}")"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Dry run for $run_name -- no job submitted."
    else
        "${SBATCH_CMD[@]}"
    fi
    echo ""
done

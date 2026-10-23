#!/usr/bin/env bash
set -euo pipefail

# Submit wrapper for jobs/run_fof_classify.sh.
# Submits one SLURM job per (run_name, buffer, score_thresh, nms_thresh) combo.
# ./submit_run_fof_classify.sh --run-names "lsst5_30k_4h200_bs192_ep50 clip5_30k_4h200_bs64_ep50" --dataset-preset test_8k --score-thresholds "0.4 0.5" --nms-thresholds "0.55 0.6"
# ./submit_run_fof_classify.sh --run-names "clip5_30k_4h200_bs64_ep50" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --buffers "1"
# ./submit_run_fof_classify.sh --run-names "lsst5_all_4h200_bs192_ep20" --dataset-preset test_all --score-thresholds "0.45 0.55" --nms-thresholds "0.65 0.65"
# ./submit_run_fof_classify.sh --run-names "clip5_flatten_30k_4h200_bs64_ep15_resume" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --buffers "1 2"

# ./submit_run_fof_classify.sh --run-names "clip5_flatten_30k_4h200_bs64_ep15 clip5_30k_4h200_bs192_ep15" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --buffers "1"
# ./submit_run_fof_classify.sh --run-names "clip5_flatten_30k_4h200_bs64_ep15_resume clip5_30k_4h200_bs64_ep50" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --buffers "1"

# ./submit_run_fof_classify.sh --run-names "comb_30k_4h200_bs144_ep50" --dataset-preset test_8k --score-thresholds "0.4 0.55 0.65" --nms-thresholds "0.55 0.65 0.65" --buffers "1 2"
# ./submit_run_fof_classify.sh --run-names "comb_30k_4h200_bs144_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65" --buffers "1"
# ./submit_run_fof_classify.sh --run-names "comb_30k_4h200_bs144_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65" --buffers "1"  --skip-lsst

# ./submit_run_fof_classify.sh -j paper_comb --run-names "comb_30k_4h200_bs144_ep50" --dataset-preset test_8k --score-thresholds "0.55" --nms-thresholds "0.65"
# ./submit_run_fof_classify.sh -j paper_comb --run-names "comb_30k_4h200_bs144_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.7"
# ./submit_run_fof_classify.sh -j paper_base --run-names "lsst5_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.4 0.5" --nms-thresholds "0.55 0.6"
# ./submit_run_fof_classify.sh -j paper_ups --run-names "lsst5_ups_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65"
# ./jobs/submit_run_fof_classify.sh -j paper_base_ms --run-names "lsst5_30k_4h200_bs192_ep50_musigma" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65"
# ./jobs/submit_run_fof_classify.sh -j paper_base_pad --run-names "lsst5_pad_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65"
# ./jobs/submit_run_fof_classify.sh -j paper_base_ps13 --run-names "lsst5_ps13_ups_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65"
               
# ./submit_run_fof_classify.sh --run-names "lsst5_30k_4h200_bs192_ep50 distill_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65" --buffers "1"
# ./submit_run_fof_classify.sh --run-names "lsst5_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65" --buffers "1"

# ./submit_run_fof_classify.sh --run-names "clip5_30k_4h200_bs192_ep15_lprj" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --buffers "1"
# ./submit_run_fof_classify.sh --run-names "lsst5_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --buffers "1,2"

# ~/jobs/submit_run_fof_classify.sh --dataset-preset test_8k --model-specs "lsst5_30k_4h200_bs192_ep50=0.5:0.6" --model-specs "lsst5_ups_30k_4h200_bs192_ep50=0.6:0.65" --model-specs "comb_30k_4h200_bs144_ep50=0.6:0.7" --job-names "lsst5_30k_4h200_bs192_ep50=pap_tbase" --job-names "lsst5_ups_30k_4h200_bs192_ep50=pap_tups" --job-names "comb_30k_4h200_bs144_ep50=pap_tcomb"
# ``~/jobs/submit_run_fof_classify.sh -j pap_tbase_ms --run-names "lsst5_30k_4h200_bs192_ep50_musigma" --dataset-preset test_8k --score-thresholds "0.5" --nms-thresholds "0.6"``
DEFAULT_JOB_SCRIPT="$HOME/jobs/run_fof_classify.sh"

# SLURM defaults
DEFAULT_JOB_NAME="fof_cls"
DEFAULT_LOG_DIR="/projects/bfhm/yse2/logs/fof_classify"
DEFAULT_ACCOUNT="bdsp-delta-cpu"
DEFAULT_PARTITION="cpu"
DEFAULT_EXCLUDE="cn001,cn002,cn003"
DEFAULT_CONSTRAINT="work"
DEFAULT_NODES=1
DEFAULT_NTASKS_PER_NODE=1
DEFAULT_CPUS_PER_TASK=16
DEFAULT_MEM="32G"
DEFAULT_TIME="00:30:00"
DEFAULT_MAIL_USER="yse2@illinois.edu"
DEFAULT_MAIL_TYPE="ALL"

# Workflow defaults
DEFAULT_ROOT_RUN_DIR="~/lsst_runs"
DEFAULT_RUN_NAMES="lsst5_30k_4h200_bs192_ep50"
DEFAULT_DATASET_PRESET="val_4k"
DEFAULT_PREDS_DIR=""
DEFAULT_TEST_CATS_DIR=""
DEFAULT_SCORE_THRESHOLDS="0.4 0.55"
DEFAULT_NMS_THRESHOLDS="0.5 0.6"
DEFAULT_MAG_LIMIT="gold"
DEFAULT_BUFFERS="2"
DEFAULT_LINKING_LENGTHS="1.0 0.5"
DEFAULT_DMAG_SIG="2.0"

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --run-names "R1 R2 ..."       Space-separated run names (default: "$DEFAULT_RUN_NAMES")
    --model-specs "M1=s1:n1,..."  Per-model combos: "run=score:nms,..." ... (overrides --run-names/--score-thresholds/--nms-thresholds)

Workflow Options:
    --dataset-preset NAME          val_4k | val_all | test_8k | test_all | custom (default: $DEFAULT_DATASET_PRESET)
    --root-run-dir PATH            Root run directory (default: $DEFAULT_ROOT_RUN_DIR)
    --preds-dir NAME               Subdirectory under preds/. Default: eval for val_4k/val_all, else test
    --test-cats-dir PATH           Explicit test catalogs directory (required for preset=custom)
    --score-thresholds "S1 S2"     Space-separated score thresholds (default: "$DEFAULT_SCORE_THRESHOLDS")
    --nms-thresholds "N1 N2"       Space-separated NMS thresholds (default: "$DEFAULT_NMS_THRESHOLDS")
    --combo                        Cartesian product of score x NMS thresholds (default: paired by index)
    --mag-limit NAME               gold | power_law | nominal (default: $DEFAULT_MAG_LIMIT)
    --buffers "B1 B2"              Space-separated buffers in [0,1,2] (default: "$DEFAULT_BUFFERS")
    --linking-lengths "L1 L2"      Space-separated linking lengths (default: "$DEFAULT_LINKING_LENGTHS")
    --dmag-sig FLOAT               Significant-blend mag gap threshold (default: $DEFAULT_DMAG_SIG)
    --match-rad FLOAT              Optional stage2 match radius override
    --job-names "run1=custom1"     Per-run job name prefix override (repeatable, model-specs mode)
    --skip-lsst                    Pass --skip-lsst to run_fof_classify.py

SLURM Options:
    -s, --script PATH              Job script to submit (default: $DEFAULT_JOB_SCRIPT)
    -j, --job-name NAME            Base SLURM job name (default: $DEFAULT_JOB_NAME)
        --log-dir PATH             Directory for stdout/stderr logs (default: $DEFAULT_LOG_DIR)
    -A, --account NAME             SLURM account (default: $DEFAULT_ACCOUNT)
    -p, --partition NAME           SLURM partition (default: $DEFAULT_PARTITION)
        --exclude HOSTLIST         Nodes to exclude (default: $DEFAULT_EXCLUDE)
        --constraint EXPR          Node feature constraint (default: $DEFAULT_CONSTRAINT)
        --nodes N                  Number of nodes (default: $DEFAULT_NODES)
        --ntasks-per-node N        Tasks per node (default: $DEFAULT_NTASKS_PER_NODE)
    -c, --cpus N                   CPUs per task (default: $DEFAULT_CPUS_PER_TASK)
    -m, --mem SIZE                 Memory allocation (default: $DEFAULT_MEM)
    -t, --time HH:MM:SS            Time limit (default: $DEFAULT_TIME)
        --mail-user EMAIL          Email for notifications (default: $DEFAULT_MAIL_USER)
        --mail-type TYPE           Notification events (default: $DEFAULT_MAIL_TYPE)

Misc:
    --dry-run                      Print sbatch commands without submitting
    -h, --help                     Show this message

Examples:
    $0 --run-names "lsst5_30k_4h200_bs192_ep50 clip5_30k_4h200_bs64_ep50" \
       --dataset-preset test_8k --score-thresholds "0.4 0.5" --nms-thresholds "0.55 0.6" \
       --mag-limit gold --buffers "1 2" --dry-run

    $0 --run-names "lsst5_all_4h200_bs192_ep20" --dataset-preset test_all \
       --score-thresholds "0.45 0.55" --nms-thresholds "0.65 0.65"

    $0 --dataset-preset test_8k --buffers "1 2" \
       --model-specs "lsst5_30k_4h200_bs192_ep50=0.4:0.55,0.5:0.6" --model-specs "clip5_30k_4h200_bs64_ep50=0.55:0.65"

    $0 --dataset-preset test_8k --buffers "2" \
       --model-specs "lsst5_30k_4h200_bs192_ep50=0.5:0.6" --model-specs "lsst5_ups_30k_4h200_bs192_ep50=0.6:0.65" \
       --job-names "lsst5_30k_4h200_bs192_ep50=paper_base" --job-names "lsst5_ups_30k_4h200_bs192_ep50=paper_ups"
EOF
}

# Defaults
JOB_SCRIPT="$DEFAULT_JOB_SCRIPT"
JOB_NAME="$DEFAULT_JOB_NAME"
LOG_DIR="$DEFAULT_LOG_DIR"
ACCOUNT="$DEFAULT_ACCOUNT"
PARTITION="$DEFAULT_PARTITION"
EXCLUDE="$DEFAULT_EXCLUDE"
CONSTRAINT="$DEFAULT_CONSTRAINT"
NODES="$DEFAULT_NODES"
NTASKS_PER_NODE="$DEFAULT_NTASKS_PER_NODE"
CPUS="$DEFAULT_CPUS_PER_TASK"
MEMORY="$DEFAULT_MEM"
TIME_LIMIT="$DEFAULT_TIME"
MAIL_USER="$DEFAULT_MAIL_USER"
MAIL_TYPE="$DEFAULT_MAIL_TYPE"

ROOT_RUN_DIR="$DEFAULT_ROOT_RUN_DIR"
DATASET_PRESET="$DEFAULT_DATASET_PRESET"
PREDS_DIR="$DEFAULT_PREDS_DIR"
TEST_CATS_DIR="$DEFAULT_TEST_CATS_DIR"
RUN_NAMES_RAW="$DEFAULT_RUN_NAMES"
SCORE_THRESHOLDS="$DEFAULT_SCORE_THRESHOLDS"
NMS_THRESHOLDS="$DEFAULT_NMS_THRESHOLDS"
COMBO="false"
MAG_LIMIT="$DEFAULT_MAG_LIMIT"
BUFFERS_RAW="$DEFAULT_BUFFERS"
LINKING_LENGTHS="$DEFAULT_LINKING_LENGTHS"
DMAG_SIG="$DEFAULT_DMAG_SIG"
MATCH_RAD=""
SKIP_LSST="false"
DRY_RUN="false"
MODEL_SPECS=()
JOB_NAMES_SPECS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--script)            JOB_SCRIPT="$2";           shift 2 ;;
        -j|--job-name)          JOB_NAME="$2";             shift 2 ;;
        --log-dir)              LOG_DIR="$2";              shift 2 ;;
        -A|--account)           ACCOUNT="$2";              shift 2 ;;
        -p|--partition)         PARTITION="$2";            shift 2 ;;
        --exclude)              EXCLUDE="$2";              shift 2 ;;
        --constraint)           CONSTRAINT="$2";           shift 2 ;;
        --nodes)                NODES="$2";                shift 2 ;;
        --ntasks-per-node)      NTASKS_PER_NODE="$2";      shift 2 ;;
        -c|--cpus)              CPUS="$2";                 shift 2 ;;
        -m|--mem)               MEMORY="$2";               shift 2 ;;
        -t|--time)              TIME_LIMIT="$2";           shift 2 ;;
        --mail-user)            MAIL_USER="$2";            shift 2 ;;
        --mail-type)            MAIL_TYPE="$2";            shift 2 ;;

        --dataset-preset)       DATASET_PRESET="$2";       shift 2 ;;
        --root-run-dir)         ROOT_RUN_DIR="$2";         shift 2 ;;
        --preds-dir)            PREDS_DIR="$2";            shift 2 ;;
        --test-cats-dir)        TEST_CATS_DIR="$2";        shift 2 ;;
        --run-names)            RUN_NAMES_RAW="$2";        shift 2 ;;
        --score-thresholds)     SCORE_THRESHOLDS="$2";     shift 2 ;;
        --nms-thresholds)       NMS_THRESHOLDS="$2";       shift 2 ;;
        --mag-limit)            MAG_LIMIT="$2";            shift 2 ;;
        --buffers)              BUFFERS_RAW="$2";          shift 2 ;;
        --linking-lengths)      LINKING_LENGTHS="$2";      shift 2 ;;
        --dmag-sig)             DMAG_SIG="$2";             shift 2 ;;
        --match-rad)            MATCH_RAD="$2";            shift 2 ;;
        --combo)                COMBO="true";             shift ;;
        --model-specs)          MODEL_SPECS+=("$2");       shift 2 ;;
        --job-names)            JOB_NAMES_SPECS+=("$2");   shift 2 ;;
        --skip-lsst)            SKIP_LSST="true";          shift ;;
        --dry-run)              DRY_RUN="true";            shift ;;
        -h|--help)              show_usage; exit 0 ;;
        *)
            echo "Error: Unknown option '$1'"
            show_usage
            exit 1
            ;;
    esac
done

case "$DATASET_PRESET" in
    val_4k)
        [[ -z "$TEST_CATS_DIR" ]] && TEST_CATS_DIR="~/lsst_data/test_cats_lvl5/val_4k/"
        [[ -z "$PREDS_DIR" ]] && PREDS_DIR="eval"
        ;;
    val_all)
        [[ -z "$TEST_CATS_DIR" ]] && TEST_CATS_DIR="~/lsst_data/test_cats_lvl5/val_all/"
        [[ -z "$PREDS_DIR" ]] && PREDS_DIR="eval"
        ;;
    test_8k)
        [[ -z "$TEST_CATS_DIR" ]] && TEST_CATS_DIR="~/lsst_data/test_cats_lvl5/test_8k/"
        [[ -z "$PREDS_DIR" ]] && PREDS_DIR="test"
        ;;
    test_all)
        [[ -z "$TEST_CATS_DIR" ]] && TEST_CATS_DIR="~/lsst_data/test_cats_lvl5/test_all/"
        [[ -z "$PREDS_DIR" ]] && PREDS_DIR="test"
        ;;
    custom)
        [[ -z "$TEST_CATS_DIR" ]] && {
            echo "Error: --test-cats-dir is required when --dataset-preset custom"
            exit 1
        }
        [[ -z "$PREDS_DIR" ]] && PREDS_DIR="test"
        ;;
    *)
        echo "Error: --dataset-preset must be one of: val_4k, val_all, test_8k, test_all, eval, custom"
        exit 1
        ;;
esac

if [[ ! -f "$JOB_SCRIPT" ]]; then
    echo "Error: Job script not found: $JOB_SCRIPT"
    exit 1
fi
if [[ ! "$MAG_LIMIT" =~ ^(gold|power_law|nominal)$ ]]; then
    echo "Error: --mag-limit must be one of: gold, power_law, nominal"
    exit 1
fi

ROOT_RUN_DIR_EXPANDED="${ROOT_RUN_DIR/#\~/$HOME}"
TEST_CATS_DIR_EXPANDED="${TEST_CATS_DIR/#\~/$HOME}"

if [[ ! -d "$ROOT_RUN_DIR_EXPANDED" ]]; then
    echo "Error: root run dir does not exist: $ROOT_RUN_DIR_EXPANDED"
    exit 1
fi
if [[ ! -d "$TEST_CATS_DIR_EXPANDED" ]]; then
    echo "Error: test cats dir does not exist: $TEST_CATS_DIR_EXPANDED"
    exit 1
fi

read -ra BUFFERS_ARR <<< "$BUFFERS_RAW"
read -ra LINKING_LENGTHS_ARR <<< "$LINKING_LENGTHS"

if [[ ${#MODEL_SPECS[@]} -gt 0 ]]; then
    # Model-specs mode: skip legacy run-names/thresholds validation
    :
else
    read -ra RUN_NAMES <<< "$RUN_NAMES_RAW"
    read -ra SCORE_THRESH_ARR <<< "$SCORE_THRESHOLDS"
    read -ra NMS_THRESH_ARR <<< "$NMS_THRESHOLDS"

    if [[ "${#RUN_NAMES[@]}" -eq 0 ]]; then
        echo "Error: --run-names did not contain any run names"
        exit 1
    fi
    if [[ "${#SCORE_THRESH_ARR[@]}" -eq 0 ]] || [[ "${#NMS_THRESH_ARR[@]}" -eq 0 ]]; then
        echo "Error: threshold lists cannot be empty"
        exit 1
    fi
fi
if [[ "${#BUFFERS_ARR[@]}" -eq 0 ]]; then
    echo "Error: --buffers cannot be empty"
    exit 1
fi
if [[ "${#LINKING_LENGTHS_ARR[@]}" -eq 0 ]]; then
    echo "Error: --linking-lengths cannot be empty"
    exit 1
fi

for b in "${BUFFERS_ARR[@]}"; do
    if [[ ! "$b" =~ ^[0-2]$ ]]; then
        echo "Error: buffer must be one of 0,1,2 (got '$b')"
        exit 1
    fi
done
if [[ ${#MODEL_SPECS[@]} -eq 0 ]]; then
    for s in "${SCORE_THRESH_ARR[@]}"; do
        awk -v x="$s" 'BEGIN{exit !(x+0>=0 && x+0<=1)}' || {
            echo "Error: score threshold out of range [0,1]: $s"
            exit 1
        }
    done
    for n in "${NMS_THRESH_ARR[@]}"; do
        awk -v x="$n" 'BEGIN{exit !(x+0>=0 && x+0<=1)}' || {
            echo "Error: nms threshold out of range [0,1]: $n"
            exit 1
        }
    done
fi
for ll in "${LINKING_LENGTHS_ARR[@]}"; do
    awk -v x="$ll" 'BEGIN{exit !(x+0>0)}' || {
        echo "Error: linking length must be > 0: $ll"
        exit 1
    }
done

mkdir -p "$LOG_DIR"

# Build job-names map from --job-names specs
declare -A JOB_NAMES_MAP
for spec in "${JOB_NAMES_SPECS[@]}"; do
    run="${spec%%=*}"
    name="${spec#*=}"
    if [[ -z "$run" ]] || [[ -z "$name" ]]; then
        echo "Error: invalid --job-names entry (expected run=name): $spec"
        exit 1
    fi
    JOB_NAMES_MAP["$run"]="$name"
done

echo "FOF classify configuration:"
if [[ ${#MODEL_SPECS[@]} -gt 0 ]]; then
    echo "  Mode              : model-specs (per-model combos)"
    echo "  Model specs       : ${MODEL_SPECS[*]}"
else
    echo "  Run names         : ${RUN_NAMES[*]}"
    echo "  Score thresholds  : $SCORE_THRESHOLDS"
    echo "  NMS thresholds    : $NMS_THRESHOLDS"
fi
echo "  Root run dir      : $ROOT_RUN_DIR_EXPANDED"
echo "  Dataset preset    : $DATASET_PRESET"
echo "  Test cats dir     : $TEST_CATS_DIR_EXPANDED"
echo "  Preds dir         : $PREDS_DIR"
echo "  Mag limit         : $MAG_LIMIT"
echo "  Buffers           : $BUFFERS_RAW"
echo "  Linking lengths   : $LINKING_LENGTHS"
echo "  DMag sig          : ${DMAG_SIG:-<none>}"
echo "  Match rad         : ${MATCH_RAD:-<per-linking-length>}"
echo "  Skip LSST         : $SKIP_LSST"
if [[ ${#JOB_NAMES_SPECS[@]} -gt 0 ]]; then
    echo "  Job names         : ${JOB_NAMES_SPECS[*]}"
fi
echo ""
echo "SLURM configuration:"
echo "  Base job name     : $JOB_NAME"
echo "  Log dir           : $LOG_DIR"
echo "  Account           : $ACCOUNT"
echo "  Partition         : $PARTITION"
echo "  Exclude           : ${EXCLUDE:-<none>}"
echo "  Constraint        : $CONSTRAINT"
echo "  Nodes             : $NODES"
echo "  Tasks/node        : $NTASKS_PER_NODE"
echo "  CPUs per task     : $CPUS"
echo "  Memory            : $MEMORY"
echo "  Time              : $TIME_LIMIT"
echo "  Mail user         : $MAIL_USER  (type: $MAIL_TYPE)"
echo ""

get_truth_mag_limit() {
    local mag_limit="$1"
    local buffer="$2"
    python - "$mag_limit" "$buffer" << 'PY'
        import sys
        mag = sys.argv[1]
        buffer = int(sys.argv[2])
        base = {
            'power_law': 26.22,
            'gold': 25.3,
            'nominal': 26.42,
        }[mag]
        print(f"{base + buffer:.2f}")
PY
}

# --- Submit helper (shared by both paths) ---
submit_one() {
    local run_name="$1" buffer="$2" score="$3" nms="$4"
    local safe_score="${score//./p}"
    local safe_nms="${nms//./p}"
    local custom_prefix="${JOB_NAMES_MAP[$run_name]:-${JOB_NAME}_${run_name}}"
    local job_name_run="${custom_prefix}_buf${buffer}_s${safe_score}_n${safe_nms}"
    export ROOT_RUN_DIR="$ROOT_RUN_DIR_EXPANDED"
    export RUN_NAME="$run_name"
    export TEST_CATS_DIR="$TEST_CATS_DIR_EXPANDED"
    export PREDS_DIR
    export SCORE_THRESH="$score"
    export NMS_THRESH="$nms"
    export MAG_LIMIT
    export BUFFER="$buffer"
    export LINKING_LENGTHS
    export DMAG_SIG
    export MATCH_RAD
    export SKIP_LSST

    SBATCH_CMD=(
        sbatch
        "--job-name=$job_name_run"
        "--output=${LOG_DIR}/%x.%j.%N.out"
        "--error=${LOG_DIR}/%x.%j.%N.err"
        "--account=$ACCOUNT"
        "--partition=$PARTITION"
        "--exclude=$EXCLUDE"
        "--constraint=$CONSTRAINT"
        "--nodes=$NODES"
        "--ntasks-per-node=$NTASKS_PER_NODE"
        "--cpus-per-task=$CPUS"
        "--mem=$MEMORY"
        "--time=$TIME_LIMIT"
        "--mail-user=$MAIL_USER"
        "--mail-type=$MAIL_TYPE"
        "$JOB_SCRIPT"
    )
    echo "Command:$(printf ' %q' "${SBATCH_CMD[@]}")"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "Dry run for run=$run_name buffer=$buffer score=$score nms=$nms -- no job submitted."
    else
        "${SBATCH_CMD[@]}"
    fi
    echo ""
}

if [[ ${#MODEL_SPECS[@]} -gt 0 ]]; then
    # --- Model-specs mode: per-model combos ---
    echo "Model specs:"
    declare -A MODEL_COMBOS
    MODEL_NAMES=()
    for spec in "${MODEL_SPECS[@]}"; do
        run_name="${spec%%=*}"
        combos_str="${spec#*=}"
        if [[ -z "$run_name" ]] || [[ -z "$combos_str" ]]; then
            echo "Error: invalid --model-specs entry (expected run_name=score:nms,...): $spec"
            exit 1
        fi
        run_dir="$ROOT_RUN_DIR_EXPANDED/$run_name"
        if [[ ! -d "$run_dir" ]]; then
            echo "Error: run directory missing for model spec: $run_dir"
            exit 1
        fi
        IFS=',' read -ra PAIRS <<< "$combos_str"
        validated_pairs=()
        for pair in "${PAIRS[@]}"; do
            score="${pair%%:*}"
            nms="${pair##*:}"
            if [[ -z "$score" ]] || [[ -z "$nms" ]]; then
                echo "Error: invalid combo in --model-specs (expected score:nms): $pair"
                exit 1
            fi
            awk -v x="$score" 'BEGIN{exit !(x+0>=0 && x+0<=1)}' || {
                echo "Error: score threshold out of range [0,1]: $score"
                exit 1
            }
            awk -v x="$nms" 'BEGIN{exit !(x+0>=0 && x+0<=1)}' || {
                echo "Error: nms threshold out of range [0,1]: $nms"
                exit 1
            }
            validated_pairs+=("${score}:${nms}")
        done
        MODEL_COMBOS["$run_name"]="${validated_pairs[*]}"
        MODEL_NAMES+=("$run_name")
        echo "  ${run_name}: ${validated_pairs[*]}"
    done
    echo ""

    for run_name in "${MODEL_NAMES[@]}"; do
        run_dir="$ROOT_RUN_DIR_EXPANDED/$run_name"
        IFS=' ' read -ra COMBOS_ARR <<< "${MODEL_COMBOS[$run_name]}"
        for buffer in "${BUFFERS_ARR[@]}"; do
            total="${#COMBOS_ARR[@]}"
            pending="$total"
            for pair in "${COMBOS_ARR[@]}"; do
                score="${pair%%:*}"
                nms="${pair##*:}"
                submit_one "$run_name" "$buffer" "$score" "$nms"
            done
            echo "Run=$run_name buffer=$buffer total=$total pending=$pending"
        done
    done
else
    # --- Legacy mode: run-names × thresholds ---
    build_combos() {
        local -n out_ref=$1
        out_ref=()
        if [[ "$COMBO" == "true" ]]; then
            for s in "${SCORE_THRESH_ARR[@]}"; do
                for n in "${NMS_THRESH_ARR[@]}"; do
                    out_ref+=("${s}:${n}")
                done
            done
        else
            if [[ "${#SCORE_THRESH_ARR[@]}" -ne "${#NMS_THRESH_ARR[@]}" ]]; then
                echo "Error: --score-thresholds and --nms-thresholds must have the same length for index pairing (got ${#SCORE_THRESH_ARR[@]} vs ${#NMS_THRESH_ARR[@]}). Use --combo for cartesian product."
                exit 1
            fi
            for i in "${!SCORE_THRESH_ARR[@]}"; do
                out_ref+=("${SCORE_THRESH_ARR[$i]}:${NMS_THRESH_ARR[$i]}")
            done
        fi
    }

    COMBOS=()
    build_combos COMBOS

    for run_name in "${RUN_NAMES[@]}"; do
        run_dir="$ROOT_RUN_DIR_EXPANDED/$run_name"
        if [[ ! -d "$run_dir" ]]; then
            echo "Warning: run directory missing, skipping: $run_dir"
            continue
        fi
        for buffer in "${BUFFERS_ARR[@]}"; do
            total=0
            pending=0
            for pair in "${COMBOS[@]}"; do
                score="${pair%%:*}"
                nms="${pair##*:}"
                total=$((total + 1))
                pending=$((pending + 1))
                submit_one "$run_name" "$buffer" "$score" "$nms"
            done
            echo "Run=$run_name buffer=$buffer total=$total pending=$pending"
            if [[ "$pending" -eq 0 ]]; then
                echo "No pending combos for run=$run_name buffer=$buffer."
                echo ""
            fi
        done
    done
fi

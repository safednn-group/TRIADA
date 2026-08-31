#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  # Run the full Car 3 x 3 grid (9 fine-tuning runs), or the six reported
  # Pedestrian cells when TRIADA_OBJECT_CLASS=pedestrian.
  #
  # Each dataset setup gets its own checkpoint:
  tools/triada/run_grid.sh \
    CHECKPOINT_WAYMO2KITTI \
    CHECKPOINT_WAYMO2NUSC \
    CHECKPOINT_NUSC2KITTI \
    [train.py args...]

Car dataset setups / budgets:
  waymo2kitti   10 37 100
  waymo2nusc    10 100 281
  nusc2kitti    10 37 100

Pedestrian setups / budgets:
  waymo2kitti   10 100
  waymo2nusc    10 100
  nusc2kitti    10 100

The script extracts NAP activations once per dataset setup, then reuses those
activation dumps for each relevant budget. Each budget is then passed through
run_best_constant_finetune.sh, which selects the NAP subset, builds dataset
infos, and launches the constant fine-tuning grid.

Environment:
  TRIADA_NUSCENES_DATA_PATH   default: data/nuscenes/v1.0-trainval
  TRIADA_NUSCENES_PKLS_PATH   default: data/nuscenes/v1.0-trainval
  TRIADA_FULL_GRID_TAG        default: full_grid_9
  TRIADA_NAP_DEVICE           default: cuda
  TRIADA_OBJECT_CLASS          car (default) or pedestrian

Notes:
  - The three checkpoint arguments may point to the same checkpoint if that is
    what your experiment requires.
  - Extra arguments after the three checkpoints are forwarded to the
    fine-tuning command for every Car (9) or Pedestrian (6) run.
  - Existing activation dumps are reused when they are found at the expected
    locations; set TRIADA_FORCE_EXTRACT=1 to regenerate them.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 3 ]]; then
    usage
    exit 2
fi

CHECKPOINT_WAYMO2KITTI=$1
CHECKPOINT_WAYMO2NUSC=$2
CHECKPOINT_NUSC2KITTI=$3
shift 3
EXTRA_ARGS=("$@")

CALLER_CWD=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)

NUSCENES_PKLS_PATH=${TRIADA_NUSCENES_PKLS_PATH:-${REPO_ROOT}/data/nuscenes/v1.0-trainval}
NUSCENES_DATA_PATH=${TRIADA_NUSCENES_DATA_PATH:-${NUSCENES_PKLS_PATH}}
RUN_TAG=${TRIADA_FULL_GRID_TAG:-full_grid_9}
OBJECT_CLASS=${TRIADA_OBJECT_CLASS:-car}

case "${OBJECT_CLASS}" in
    car)
        ACTIVATION_OUTPUT_ROOT="${REPO_ROOT}/output/triada/car/nap_activation"
        WAYMO2KITTI_BUDGETS=(10 37 100)
        WAYMO2NUSC_BUDGETS=(10 100 281)
        NUSC2KITTI_BUDGETS=(10 37 100)
        ;;
    pedestrian)
        ACTIVATION_OUTPUT_ROOT="${REPO_ROOT}/output/triada/pedestrian/nap_activation"
        WAYMO2KITTI_BUDGETS=(10 100)
        WAYMO2NUSC_BUDGETS=(10 100)
        NUSC2KITTI_BUDGETS=(10 100)
        ;;
    *)
        echo "Invalid TRIADA_OBJECT_CLASS ${OBJECT_CLASS}; expected car or pedestrian" >&2
        exit 2
        ;;
esac

if [[ "${CHECKPOINT_WAYMO2KITTI}" != /* ]]; then
    CHECKPOINT_WAYMO2KITTI="${CALLER_CWD}/${CHECKPOINT_WAYMO2KITTI}"
fi
if [[ "${CHECKPOINT_WAYMO2NUSC}" != /* ]]; then
    CHECKPOINT_WAYMO2NUSC="${CALLER_CWD}/${CHECKPOINT_WAYMO2NUSC}"
fi
if [[ "${CHECKPOINT_NUSC2KITTI}" != /* ]]; then
    CHECKPOINT_NUSC2KITTI="${CALLER_CWD}/${CHECKPOINT_NUSC2KITTI}"
fi
if [[ "${NUSCENES_PKLS_PATH}" != /* ]]; then
    NUSCENES_PKLS_PATH="${CALLER_CWD}/${NUSCENES_PKLS_PATH}"
fi
if [[ "${NUSCENES_DATA_PATH}" != /* ]]; then
    NUSCENES_DATA_PATH="${CALLER_CWD}/${NUSCENES_DATA_PATH}"
fi

for CHECKPOINT in \
    "${CHECKPOINT_WAYMO2KITTI}" \
    "${CHECKPOINT_WAYMO2NUSC}" \
    "${CHECKPOINT_NUSC2KITTI}"; do
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "Missing checkpoint: ${CHECKPOINT}" >&2
        exit 1
    fi
done

for DATA_DIRECTORY in "${NUSCENES_DATA_PATH}" "${NUSCENES_PKLS_PATH}"; do
    if [[ ! -d "${DATA_DIRECTORY}" ]]; then
        echo "Missing nuScenes directory: ${DATA_DIRECTORY}" >&2
        exit 1
    fi
done

# Extract once per source/target setup.  The output names mirror
# run_best_constant_finetune.sh.
extract_setup() {
    local shift_name=$1
    local experiment=$2
    local checkpoint=$3
    local source_output_name=$4
    local target_output_name=$5

    # A checkpoint epoch is not unique across source models or dataset shifts.
    local setup_tag="${RUN_TAG}_${shift_name}"
    local checkpoint_name epoch_id
    checkpoint_name=$(basename "${checkpoint}")
    mapfile -t checkpoint_numbers < <(
        grep -oE '[0-9]+' <<< "${checkpoint_name}" || true
    )

    if [[ ${#checkpoint_numbers[@]} -gt 0 ]]; then
        epoch_id=${checkpoint_numbers[-1]}
    else
        epoch_id=no_number
    fi

    local source_result_pkl="${ACTIVATION_OUTPUT_ROOT}/${source_output_name}/${setup_tag}/eval/epoch_${epoch_id}/val/result.pkl"
    local target_result_pkl="${ACTIVATION_OUTPUT_ROOT}/${target_output_name}/${setup_tag}/eval/epoch_${epoch_id}/train/result.pkl"

    if [[ "${TRIADA_FORCE_EXTRACT:-0}" != "1" && -f "${source_result_pkl}" && -f "${target_result_pkl}" ]]; then
        echo
        echo "=== ${shift_name}: reusing existing activation dumps ==="
        echo "source: ${source_result_pkl}"
        echo "target: ${target_result_pkl}"
    else
        local source_extract_args=()
        local target_extract_args=()

        case "${shift_name}" in
            waymo2nusc)
                target_extract_args=(
                    --data_path "${NUSCENES_DATA_PATH}"
                    --pkls_path "${NUSCENES_PKLS_PATH}"
                )
                ;;
            nusc2kitti)
                source_extract_args=(
                    --data_path "${NUSCENES_DATA_PATH}"
                    --pkls_path "${NUSCENES_PKLS_PATH}"
                )
                ;;
        esac

        echo
        echo "=== ${shift_name}: extracting source activations ==="
        "${SCRIPT_DIR}/extract_nap_activations.sh" \
            "${experiment}" source "${checkpoint}" \
            --extra_tag "${setup_tag}" \
            --eval_tag "" \
            "${source_extract_args[@]}"

        echo
        echo "=== ${shift_name}: extracting target activations ==="
        "${SCRIPT_DIR}/extract_nap_activations.sh" \
            "${experiment}" target "${checkpoint}" \
            --extra_tag "${setup_tag}" \
            --eval_tag "" \
            "${target_extract_args[@]}"
    fi

    if [[ ! -f "${source_result_pkl}" ]]; then
        echo "Missing source activation dump after extraction: ${source_result_pkl}" >&2
        exit 1
    fi
    if [[ ! -f "${target_result_pkl}" ]]; then
        echo "Missing target activation dump after extraction: ${target_result_pkl}" >&2
        exit 1
    fi

    # Return the two paths to the caller through globals.
    SOURCE_RESULT_PKL="${source_result_pkl}"
    TARGET_RESULT_PKL="${target_result_pkl}"
}

run_setup() {
    local shift_name=$1
    local experiment=$2
    local checkpoint=$3
    shift 3
    local budgets=("$@")

    local source_output_name target_output_name
    case "${shift_name}" in
        waymo2kitti)
            source_output_name="source_waymo16_ground_truth"
            target_output_name="target_kitti_proposals"
            ;;
        waymo2nusc)
            source_output_name="source_waymo16_ground_truth"
            target_output_name="target_nuscenes_proposals_for_waymo16"
            ;;
        nusc2kitti)
            source_output_name="source_nuscenes_ground_truth"
            target_output_name="target_kitti_proposals"
            ;;
        *)
            echo "Internal error: unknown setup ${shift_name}" >&2
            exit 2
            ;;
    esac

    extract_setup \
        "${shift_name}" \
        "${experiment}" \
        "${checkpoint}" \
        "${source_output_name}" \
        "${target_output_name}"

    for budget in "${budgets[@]}"; do
        echo
        echo "============================================================"
        echo "RUN: ${shift_name}  budget=${budget}"
        echo "============================================================"

        TRIADA_RUN_TAG="${RUN_TAG}_${shift_name}_${budget}" \
        TRIADA_NUSCENES_DATA_PATH="${NUSCENES_DATA_PATH}" \
        TRIADA_NUSCENES_PKLS_PATH="${NUSCENES_PKLS_PATH}" \
        "${SCRIPT_DIR}/run_best_constant_finetune.sh" \
            "${shift_name}" \
            "${budget}" \
            "${checkpoint}" \
            "${SOURCE_RESULT_PKL}" \
            "${TARGET_RESULT_PKL}" \
            "${EXTRA_ARGS[@]}"
    done
}

run_setup \
    waymo2kitti \
    waymo16_to_kitti \
    "${CHECKPOINT_WAYMO2KITTI}" \
    "${WAYMO2KITTI_BUDGETS[@]}"

run_setup \
    waymo2nusc \
    waymo16_to_nuscenes \
    "${CHECKPOINT_WAYMO2NUSC}" \
    "${WAYMO2NUSC_BUDGETS[@]}"

run_setup \
    nusc2kitti \
    nuscenes_to_kitti \
    "${CHECKPOINT_NUSC2KITTI}" \
    "${NUSC2KITTI_BUDGETS[@]}"

echo
echo "============================================================"
if [[ "${OBJECT_CLASS}" == "pedestrian" ]]; then
    echo "Completed Pedestrian grid: 6 runs"
    echo "  waymo2kitti : budgets 10 100"
    echo "  waymo2nusc  : budgets 10 100"
    echo "  nusc2kitti  : budgets 10 100"
else
    echo "Completed full Car 3 x 3 grid: 9 runs"
    echo "  waymo2kitti : budgets 10 37 100"
    echo "  waymo2nusc  : budgets 10 100 281"
    echo "  nusc2kitti  : budgets 10 37 100"
fi
echo "============================================================"

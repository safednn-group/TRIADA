#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  # End-to-end: extract activations, select frames, build infos, and fine-tune
  tools/triada/run_best_constant_finetune.sh \
    SHIFT BUDGET CHECKPOINT [train.py args...]

  # Reuse activation dumps that were already extracted
  tools/triada/run_best_constant_finetune.sh \
    SHIFT BUDGET CHECKPOINT SOURCE_RESULT_PKL TARGET_RESULT_PKL \
    [train.py args...]

Shifts:
  waymo2kitti
  waymo2nusc
  nusc2kitti


Environment:
  TRIADA_NUSCENES_DATA_PATH   default: data/nuscenes/v1.0-trainval
  TRIADA_NUSCENES_PKLS_PATH   default: data/nuscenes/v1.0-trainval
  TRIADA_NUSCENES_SUBSET_PKLS_PATH
                              Pedestrian subset-info output directory
  TRIADA_RUN_TAG               default: SHIFT_BUDGET
  TRIADA_SUBSET_TAG            default: TRIADA_RUN_TAG
  TRIADA_OBJECT_CLASS          car (default) or pedestrian
  TRIADA_NAP_TOPK              override the per-cell candidate-pool default
  TRIADA_NAP_DEVICE            default: cuda (use cpu for cross-GPU stability)

For Waymo rows, CHECKPOINT must be trained on the canonical
LiDAR-Distillation modes/16^ Waymo source layout.
When supplied, SOURCE_RESULT_PKL and TARGET_RESULT_PKL must be extracted with
CHECKPOINT.
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

SHIFT=$1
BUDGET=$2
CHECKPOINT=$3
shift 3

case "${SHIFT}" in
    waymo2kitti)
        EXPERIMENT="waymo16_to_kitti"
        SOURCE_OUTPUT_NAME="source_waymo16_ground_truth"
        TARGET_OUTPUT_NAME="target_kitti_proposals"
        VALID_BUDGETS="10 37 100"
        ;;
    waymo2nusc)
        EXPERIMENT="waymo16_to_nuscenes"
        SOURCE_OUTPUT_NAME="source_waymo16_ground_truth"
        TARGET_OUTPUT_NAME="target_nuscenes_proposals_for_waymo16"
        VALID_BUDGETS="10 100 281"
        ;;
    nusc2kitti)
        EXPERIMENT="nuscenes_to_kitti"
        SOURCE_OUTPUT_NAME="source_nuscenes_ground_truth"
        TARGET_OUTPUT_NAME="target_kitti_proposals"
        VALID_BUDGETS="10 37 100"
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ " ${VALID_BUDGETS} " != *" ${BUDGET} "* ]]; then
    echo "Invalid budget ${BUDGET} for ${SHIFT}; expected one of: ${VALID_BUDGETS}" >&2
    exit 2
fi

OBJECT_CLASS=${TRIADA_OBJECT_CLASS:-car}
case "${OBJECT_CLASS}" in
    car|pedestrian)
        ;;
    *)
        echo "Invalid TRIADA_OBJECT_CLASS ${OBJECT_CLASS}; expected car or pedestrian" >&2
        exit 2
        ;;
esac

if [[ "${OBJECT_CLASS}" == "pedestrian" ]]; then
    DEFAULT_TOPK=10
else
    DEFAULT_TOPK=50
    case "${SHIFT}:${BUDGET}" in
        waymo2nusc:10)
            DEFAULT_TOPK=30
            ;;
        nusc2kitti:10)
            DEFAULT_TOPK=100
            ;;
    esac
fi

CALLER_CWD=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
NUSCENES_PKLS_PATH=${TRIADA_NUSCENES_PKLS_PATH:-${REPO_ROOT}/data/nuscenes/v1.0-trainval}
NUSCENES_DATA_PATH=${TRIADA_NUSCENES_DATA_PATH:-${NUSCENES_PKLS_PATH}}
RUN_TAG=${TRIADA_RUN_TAG:-${SHIFT}_${BUDGET}}
SUBSET_TAG=${TRIADA_SUBSET_TAG:-${RUN_TAG}}

if [[ "${SUBSET_TAG}" == *[!A-Za-z0-9._-]* || -z "${SUBSET_TAG}" ]]; then
    echo "TRIADA_SUBSET_TAG must contain only letters, numbers, '.', '_' or '-'" >&2
    exit 2
fi

if [[ "${OBJECT_CLASS}" == "pedestrian" ]]; then
    NUSCENES_SUBSET_PKLS_PATH=${TRIADA_NUSCENES_SUBSET_PKLS_PATH:-${REPO_ROOT}/data/triada/pedestrian/nuscenes/${SUBSET_TAG}}
else
    NUSCENES_SUBSET_PKLS_PATH=${TRIADA_NUSCENES_SUBSET_PKLS_PATH:-${NUSCENES_PKLS_PATH}}
fi

case "${OBJECT_CLASS}" in
    car)
        ACTIVATION_OUTPUT_ROOT="${REPO_ROOT}/output/triada/car/nap_activation"
        ;;
    pedestrian)
        ACTIVATION_OUTPUT_ROOT="${REPO_ROOT}/output/triada/pedestrian/nap_activation"
        ;;
    *)
        echo "Invalid TRIADA_OBJECT_CLASS ${OBJECT_CLASS}; expected car or pedestrian" >&2
        exit 2
        ;;
esac

if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="${CALLER_CWD}/${CHECKPOINT}"
fi

if [[ "${NUSCENES_PKLS_PATH}" != /* ]]; then
    NUSCENES_PKLS_PATH="${CALLER_CWD}/${NUSCENES_PKLS_PATH}"
fi
if [[ "${NUSCENES_DATA_PATH}" != /* ]]; then
    NUSCENES_DATA_PATH="${CALLER_CWD}/${NUSCENES_DATA_PATH}"
fi
if [[ "${NUSCENES_SUBSET_PKLS_PATH}" != /* ]]; then
    NUSCENES_SUBSET_PKLS_PATH="${CALLER_CWD}/${NUSCENES_SUBSET_PKLS_PATH}"
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

SOURCE_RESULT_PKL=
TARGET_RESULT_PKL=

if [[ $# -ge 2 && "${1}" != --* && "${2}" != --* ]]; then
    SOURCE_RESULT_PKL=$1
    TARGET_RESULT_PKL=$2
    shift 2

    if [[ "${SOURCE_RESULT_PKL}" != /* ]]; then
        SOURCE_RESULT_PKL="${CALLER_CWD}/${SOURCE_RESULT_PKL}"
    fi
    if [[ "${TARGET_RESULT_PKL}" != /* ]]; then
        TARGET_RESULT_PKL="${CALLER_CWD}/${TARGET_RESULT_PKL}"
    fi
else
    SOURCE_EXTRACT_ARGS=()
    TARGET_EXTRACT_ARGS=()

    case "${SHIFT}" in
        waymo2nusc)
            TARGET_EXTRACT_ARGS=(
                --data_path "${NUSCENES_DATA_PATH}"
                --pkls_path "${NUSCENES_PKLS_PATH}"
            )
            ;;
        nusc2kitti)
            SOURCE_EXTRACT_ARGS=(
                --data_path "${NUSCENES_DATA_PATH}"
                --pkls_path "${NUSCENES_PKLS_PATH}"
            )
            ;;
    esac

    if [[ "${SHIFT}" == "waymo2nusc" || "${SHIFT}" == "nusc2kitti" ]]; then
        for DATA_DIRECTORY in "${NUSCENES_DATA_PATH}" "${NUSCENES_PKLS_PATH}"; do
            if [[ ! -d "${DATA_DIRECTORY}" ]]; then
                echo "Missing nuScenes directory: ${DATA_DIRECTORY}" >&2
                exit 1
            fi
        done
    fi

    CHECKPOINT_NAME=$(basename "${CHECKPOINT}")
    mapfile -t CHECKPOINT_NUMBERS < <(grep -oE '[0-9]+' <<< "${CHECKPOINT_NAME}" || true)
    if [[ ${#CHECKPOINT_NUMBERS[@]} -gt 0 ]]; then
        EPOCH_ID=${CHECKPOINT_NUMBERS[-1]}
    else
        EPOCH_ID=no_number
    fi

    "${SCRIPT_DIR}/extract_nap_activations.sh" \
        "${EXPERIMENT}" source "${CHECKPOINT}" \
        --extra_tag "${RUN_TAG}" \
        --eval_tag "" \
        "${SOURCE_EXTRACT_ARGS[@]}"

    "${SCRIPT_DIR}/extract_nap_activations.sh" \
        "${EXPERIMENT}" target "${CHECKPOINT}" \
        --extra_tag "${RUN_TAG}" \
        --eval_tag "" \
        "${TARGET_EXTRACT_ARGS[@]}"

    SOURCE_RESULT_PKL="${ACTIVATION_OUTPUT_ROOT}/${SOURCE_OUTPUT_NAME}/${RUN_TAG}/eval/epoch_${EPOCH_ID}/val/result.pkl"
    TARGET_RESULT_PKL="${ACTIVATION_OUTPUT_ROOT}/${TARGET_OUTPUT_NAME}/${RUN_TAG}/eval/epoch_${EPOCH_ID}/train/result.pkl"
fi

for INPUT_PATH in "${CHECKPOINT}" "${SOURCE_RESULT_PKL}" "${TARGET_RESULT_PKL}"; do
    if [[ ! -f "${INPUT_PATH}" ]]; then
        echo "Missing input file: ${INPUT_PATH}" >&2
        exit 1
    fi
done

export TRIADA_NAP_TOPK="${TRIADA_NAP_TOPK:-${DEFAULT_TOPK}}"
export TRIADA_FINETUNE_TAG="${TRIADA_FINETUNE_TAG:-${RUN_TAG}}"
export TRIADA_SUBSET_TAG="${SUBSET_TAG}"
export TRIADA_NUSCENES_SUBSET_PKLS_PATH="${NUSCENES_SUBSET_PKLS_PATH}"

"${SCRIPT_DIR}/select_nap_frames.sh" \
    "${EXPERIMENT}" \
    "${SOURCE_RESULT_PKL}" \
    "${TARGET_RESULT_PKL}" \
    "${BUDGET}"

FINETUNE_ARGS=()
if [[ "${SHIFT}" == "waymo2nusc" ]]; then
    FINETUNE_ARGS=(
        --data_path "${NUSCENES_DATA_PATH}"
    )
fi

exec "${SCRIPT_DIR}/run_constant_finetune_grid.sh" \
    "${EXPERIMENT}" \
    "${BUDGET}" \
    "${CHECKPOINT}" \
    "${FINETUNE_ARGS[@]}" \
    "$@"

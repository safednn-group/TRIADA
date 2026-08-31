#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  tools/triada/run_constant_finetune_grid.sh EXPERIMENT BUDGET [PRETRAINED_CHECKPOINT] [train.py args...]

Experiments:
  waymo16_to_nuscenes    budgets: 10 100 281
  waymo16_to_kitti       budgets: 10 37 100
  nuscenes_to_kitti      budgets: 10 37 100

Environment:
  TRIADA_FINETUNE_TAG="repetition_1"
  TRIADA_OBJECT_CLASS="car"  # or "pedestrian"
  TRIADA_SUBSET_TAG="selection_1"  # Pedestrian selected-subset namespace
  TRIADA_NUSCENES_SUBSET_PKLS_PATH="/path/to/pedestrian/nuscenes/infos"
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 2 ]]; then
    usage
    exit 2
fi

EXPERIMENT=$1
BUDGET=$2
shift 2

CALLER_CWD=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TOOLS_DIR="${REPO_ROOT}/tools"
NUSCENES_PKLS_PATH=${TRIADA_NUSCENES_PKLS_PATH:-${REPO_ROOT}/data/nuscenes/v1.0-trainval}
NUSCENES_DATA_PATH=${TRIADA_NUSCENES_DATA_PATH:-${NUSCENES_PKLS_PATH}}
OBJECT_CLASS=${TRIADA_OBJECT_CLASS:-car}

case "${OBJECT_CLASS}" in
    car)
        CFG_ROOT="cfgs/triada/car"
        OUTPUT_NAMESPACE="triada"
        CHECKPOINT_SUFFIX=""
        ;;
    pedestrian)
        CFG_ROOT="cfgs/triada/pedestrian"
        OUTPUT_NAMESPACE="triada/pedestrian"
        CHECKPOINT_SUFFIX="_pedestrian"
        ;;
    *)
        echo "Invalid TRIADA_OBJECT_CLASS ${OBJECT_CLASS}; expected car or pedestrian" >&2
        exit 2
        ;;
esac

if [[ "${NUSCENES_DATA_PATH}" != /* ]]; then
    NUSCENES_DATA_PATH="${CALLER_CWD}/${NUSCENES_DATA_PATH}"
fi
if [[ "${NUSCENES_PKLS_PATH}" != /* ]]; then
    NUSCENES_PKLS_PATH="${CALLER_CWD}/${NUSCENES_PKLS_PATH}"
fi

case "${EXPERIMENT}" in
    waymo16_to_nuscenes)
        CFG_FILE="${CFG_ROOT}/finetune/waymo16_to_nuscenes_budget_${BUDGET}_${OBJECT_CLASS}.yaml"
        DEFAULT_CKPT="${REPO_ROOT}/checkpoints/pretrain_waymo16${CHECKPOINT_SUFFIX}.pth"
        EXTRA_ARGS=(
            --batch_size 1
            --data_path "${NUSCENES_DATA_PATH}"
        )
        VALID_BUDGETS="10 100 281"
        ;;
    waymo16_to_kitti)
        CFG_FILE="${CFG_ROOT}/finetune/waymo16_to_kitti_budget_${BUDGET}_${OBJECT_CLASS}.yaml"
        DEFAULT_CKPT="${REPO_ROOT}/checkpoints/pretrain_waymo16${CHECKPOINT_SUFFIX}.pth"
        EXTRA_ARGS=(--batch_size 1)
        VALID_BUDGETS="10 37 100"
        ;;
    nuscenes_to_kitti)
        CFG_FILE="${CFG_ROOT}/finetune/nuscenes_to_kitti_budget_${BUDGET}_${OBJECT_CLASS}.yaml"
        DEFAULT_CKPT="${REPO_ROOT}/checkpoints/pvrcnn_nuscenes_kitti_source_only${CHECKPOINT_SUFFIX}.pth"
        EXTRA_ARGS=()
        VALID_BUDGETS="10 37 100"
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ " ${VALID_BUDGETS} " != *" ${BUDGET} "* ]]; then
    echo "Invalid budget ${BUDGET} for ${EXPERIMENT}; expected one of: ${VALID_BUDGETS}" >&2
    exit 2
fi

if [[ "${EXPERIMENT}" == "waymo16_to_nuscenes" ]]; then
    for DATA_DIRECTORY in "${NUSCENES_DATA_PATH}" "${NUSCENES_PKLS_PATH}"; do
        if [[ ! -d "${DATA_DIRECTORY}" ]]; then
            echo "Missing nuScenes directory: ${DATA_DIRECTORY}" >&2
            exit 1
        fi
    done
fi

if [[ $# -gt 0 && "${1}" != --* ]]; then
    PRETRAINED_CKPT=$1
    shift
else
    PRETRAINED_CKPT="${DEFAULT_CKPT}"
fi

if [[ "${PRETRAINED_CKPT}" != /* ]]; then
    PRETRAINED_CKPT="${CALLER_CWD}/${PRETRAINED_CKPT}"
fi
if [[ ! -f "${PRETRAINED_CKPT}" ]]; then
    echo "Missing checkpoint: ${PRETRAINED_CKPT}" >&2
    exit 1
fi

FINETUNE_TAG=${TRIADA_FINETUNE_TAG:-}

if [[ -n "${FINETUNE_TAG}" && "${FINETUNE_TAG}" == *[!A-Za-z0-9._-]* ]]; then
    echo "TRIADA_FINETUNE_TAG must contain only letters, numbers, '.', '_' or '-'" >&2
    exit 2
fi

if [[ "${OBJECT_CLASS}" == "pedestrian" ]]; then
    SUBSET_TAG=${TRIADA_SUBSET_TAG:-${FINETUNE_TAG:-pedestrian}}
    if [[ "${SUBSET_TAG}" == *[!A-Za-z0-9._-]* || -z "${SUBSET_TAG}" ]]; then
        echo "TRIADA_SUBSET_TAG must contain only letters, numbers, '.', '_' or '-'" >&2
        exit 2
    fi

    case "${EXPERIMENT}" in
        waymo16_to_nuscenes)
            NUSCENES_FINETUNE_PKLS_PATH=${TRIADA_NUSCENES_SUBSET_PKLS_PATH:-${REPO_ROOT}/data/triada/pedestrian/nuscenes/${SUBSET_TAG}}
            if [[ "${NUSCENES_FINETUNE_PKLS_PATH}" != /* ]]; then
                NUSCENES_FINETUNE_PKLS_PATH="${CALLER_CWD}/${NUSCENES_FINETUNE_PKLS_PATH}"
            fi
            if [[ ! -d "${NUSCENES_FINETUNE_PKLS_PATH}" ]]; then
                echo "Missing Pedestrian nuScenes subset-info directory: ${NUSCENES_FINETUNE_PKLS_PATH}" >&2
                exit 1
            fi
            EXTRA_ARGS+=(--pkls_path "${NUSCENES_FINETUNE_PKLS_PATH}")
            ;;
        waymo16_to_kitti)
            PEDESTRIAN_KITTI_DATA_PATH="${REPO_ROOT}/data/triada/pedestrian/kitti/${SUBSET_TAG}/waymo16_to_kitti/budget_${BUDGET}"
            if [[ ! -d "${PEDESTRIAN_KITTI_DATA_PATH}" ]]; then
                echo "Missing Pedestrian KITTI subset directory: ${PEDESTRIAN_KITTI_DATA_PATH}" >&2
                exit 1
            fi
            EXTRA_ARGS+=(--data_path "${PEDESTRIAN_KITTI_DATA_PATH}")
            ;;
        nuscenes_to_kitti)
            PEDESTRIAN_KITTI_DATA_PATH="${REPO_ROOT}/data/triada/pedestrian/kitti/${SUBSET_TAG}/nuscenes_to_kitti/budget_${BUDGET}"
            if [[ ! -d "${PEDESTRIAN_KITTI_DATA_PATH}" ]]; then
                echo "Missing Pedestrian KITTI subset directory: ${PEDESTRIAN_KITTI_DATA_PATH}" >&2
                exit 1
            fi
            EXTRA_ARGS+=(--data_path "${PEDESTRIAN_KITTI_DATA_PATH}")
            ;;
    esac
elif [[ "${EXPERIMENT}" == "waymo16_to_nuscenes" ]]; then
    EXTRA_ARGS+=(--pkls_path "${NUSCENES_PKLS_PATH}")
fi

cd "${TOOLS_DIR}"
export PYTHONPATH="${REPO_ROOT}:${TOOLS_DIR}:${PYTHONPATH:-}"

if [[ ! -f "${CFG_FILE}" ]]; then
    echo "Missing config: ${CFG_FILE}" >&2
    exit 1
fi

RUN_BASE="../output/${OUTPUT_NAMESPACE}/${EXPERIMENT}"
if [[ -n "${FINETUNE_TAG}" ]]; then
    RUN_BASE="${RUN_BASE}/${FINETUNE_TAG}"
fi
RUN_DIR="${RUN_BASE}/budget_${BUDGET}/const_lr"

python train.py \
    --cfg_file "${CFG_FILE}" \
    --pretrained_model "${PRETRAINED_CKPT}" \
    --max_ckpt_save_num 10 \
    --output_path "${RUN_DIR}" \
    "${EXTRA_ARGS[@]}" \
    "$@"

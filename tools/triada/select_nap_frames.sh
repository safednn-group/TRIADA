#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  tools/triada/select_nap_frames.sh EXPERIMENT SOURCE_DETECTIONS TARGET_DETECTIONS [BUDGET ...]

Experiments:
  waymo16_to_nuscenes    budgets: 10 100 281
  waymo16_to_kitti       budgets: 10 37 100
  nuscenes_to_kitti      budgets: 10 37 100

SOURCE_DETECTIONS and TARGET_DETECTIONS are result.pkl files produced by extract_nap_activations.sh.

Environment:
  TRIADA_NAP_DEVICE   default: cuda (use cpu for cross-GPU-stable selection)
  TRIADA_OBJECT_CLASS car (default) or pedestrian
  TRIADA_NAP_TOPK     candidate-frame pool size
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

EXPERIMENT=$1
SOURCE_DETECTIONS=$2
TARGET_DETECTIONS=$3
shift 3

CALLER_CWD=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TOOLS_DIR="${REPO_ROOT}/tools"
OUT_ROOT="${TOOLS_DIR}/triada_subsets"
NUSCENES_PKLS_PATH=${TRIADA_NUSCENES_PKLS_PATH:-${REPO_ROOT}/data/nuscenes/v1.0-trainval}
NUSCENES_DATA_PATH=${TRIADA_NUSCENES_DATA_PATH:-${NUSCENES_PKLS_PATH}}
NAP_DEVICE=${TRIADA_NAP_DEVICE:-cuda}
TOPK_OVERRIDE=${TRIADA_NAP_TOPK:-}
OBJECT_CLASS=${TRIADA_OBJECT_CLASS:-car}

case "${OBJECT_CLASS}" in
    car|pedestrian)
        ;;
    *)
        echo "Invalid TRIADA_OBJECT_CLASS ${OBJECT_CLASS}; expected car or pedestrian" >&2
        exit 2
        ;;
esac

SUBSET_TAG=
if [[ "${OBJECT_CLASS}" == "pedestrian" ]]; then
    SUBSET_TAG=${TRIADA_SUBSET_TAG:-${TRIADA_FINETUNE_TAG:-pedestrian}}
    if [[ "${SUBSET_TAG}" == *[!A-Za-z0-9._-]* || -z "${SUBSET_TAG}" ]]; then
        echo "TRIADA_SUBSET_TAG must contain only letters, numbers, '.', '_' or '-'" >&2
        exit 2
    fi
    OUT_ROOT="${OUT_ROOT}/pedestrian/${SUBSET_TAG}"
    NUSCENES_SUBSET_PKLS_PATH=${TRIADA_NUSCENES_SUBSET_PKLS_PATH:-${REPO_ROOT}/data/triada/pedestrian/nuscenes/${SUBSET_TAG}}
else
    NUSCENES_SUBSET_PKLS_PATH=${TRIADA_NUSCENES_SUBSET_PKLS_PATH:-${NUSCENES_PKLS_PATH}}
fi

if [[ "${SOURCE_DETECTIONS}" != /* ]]; then
    SOURCE_DETECTIONS="${CALLER_CWD}/${SOURCE_DETECTIONS}"
fi

if [[ "${TARGET_DETECTIONS}" != /* ]]; then
    TARGET_DETECTIONS="${CALLER_CWD}/${TARGET_DETECTIONS}"
fi

if [[ "${NUSCENES_DATA_PATH}" != /* ]]; then
    NUSCENES_DATA_PATH="${CALLER_CWD}/${NUSCENES_DATA_PATH}"
fi
if [[ "${NUSCENES_PKLS_PATH}" != /* ]]; then
    NUSCENES_PKLS_PATH="${CALLER_CWD}/${NUSCENES_PKLS_PATH}"
fi
if [[ "${NUSCENES_SUBSET_PKLS_PATH}" != /* ]]; then
    NUSCENES_SUBSET_PKLS_PATH="${CALLER_CWD}/${NUSCENES_SUBSET_PKLS_PATH}"
fi

for INPUT_PATH in "${SOURCE_DETECTIONS}" "${TARGET_DETECTIONS}"; do
    if [[ ! -f "${INPUT_PATH}" ]]; then
        echo "Missing activation dump: ${INPUT_PATH}" >&2
        exit 1
    fi
done

case "${EXPERIMENT}" in
    waymo16_to_nuscenes)
        DEFAULT_BUDGETS=(10 100 281)
        DATASET_KIND="nuscenes"
        ;;
    waymo16_to_kitti)
        DEFAULT_BUDGETS=(10 37 100)
        DATASET_KIND="kitti"
        KITTI_SUBDIR="waymo16_to_kitti"
        ;;
    nuscenes_to_kitti)
        DEFAULT_BUDGETS=(10 37 100)
        DATASET_KIND="kitti"
        KITTI_SUBDIR="nuscenes_to_kitti"
        ;;
    *)
        usage
        exit 2
        ;;
esac

if [[ $# -gt 0 ]]; then
    BUDGETS=("$@")
else
    BUDGETS=("${DEFAULT_BUDGETS[@]}")
fi

for BUDGET in "${BUDGETS[@]}"; do
    if [[ " ${DEFAULT_BUDGETS[*]} " != *" ${BUDGET} "* ]]; then
        echo "Invalid budget ${BUDGET} for ${EXPERIMENT}; expected one of: ${DEFAULT_BUDGETS[*]}" >&2
        exit 2
    fi
done

if [[ "${DATASET_KIND}" == "nuscenes" ]]; then
    for DATA_PATH in \
        "${NUSCENES_DATA_PATH}" \
        "${NUSCENES_PKLS_PATH}"; do
        if [[ ! -d "${DATA_PATH}" ]]; then
            echo "Missing nuScenes directory: ${DATA_PATH}" >&2
            exit 1
        fi
    done
fi

cd "${TOOLS_DIR}"
export PYTHONPATH="${REPO_ROOT}:${TOOLS_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}"

for BUDGET in "${BUDGETS[@]}"; do
    if [[ -n "${TOPK_OVERRIDE}" ]]; then
        TOPK=${TOPK_OVERRIDE}
    elif [[ "${OBJECT_CLASS}" == "pedestrian" ]]; then
        TOPK=10
    else
        TOPK=50
        case "${EXPERIMENT}:${BUDGET}" in
            waymo16_to_nuscenes:10)
                TOPK=30
                ;;
            nuscenes_to_kitti:10)
                TOPK=100
                ;;
        esac
    fi
    if ! [[ "${TOPK}" =~ ^[1-9][0-9]*$ ]]; then
        echo "TRIADA_NAP_TOPK must be a positive integer" >&2
        exit 2
    fi

    if [[ "${OBJECT_CLASS}" == "pedestrian" ]]; then
        SUBSET_NAME="pedestrian_${EXPERIMENT}_budget_${BUDGET}"
    else
        SUBSET_NAME="${EXPERIMENT}_budget_${BUDGET}"
    fi
    SUBSET_JSON="${OUT_ROOT}/${SUBSET_NAME}.json"

    if [[ "${DATASET_KIND}" == "kitti" ]]; then
        if [[ "${OBJECT_CLASS}" == "pedestrian" ]]; then
            DATA_ROOT="../data/triada/pedestrian/kitti/${SUBSET_TAG}/${KITTI_SUBDIR}/budget_${BUDGET}"
        else
            DATA_ROOT="../data/triada/kitti/${KITTI_SUBDIR}/budget_${BUDGET}"
        fi
        IMAGESETS_ROOT="${DATA_ROOT}/ImageSets"

        case "${EXPERIMENT}" in
            waymo16_to_kitti)
                DATA_CFG="cfgs/triada/data/target_kitti_waymo16_budget_${BUDGET}.yaml"
                ;;
            nuscenes_to_kitti)
                DATA_CFG="cfgs/triada/data/target_kitti_nuscenes_budget_${BUDGET}.yaml"
                ;;
        esac

        mkdir -p "${IMAGESETS_ROOT}"
        ln -sfn "${REPO_ROOT}/data/kitti/training" "${DATA_ROOT}/training"
        ln -sfn "${REPO_ROOT}/data/kitti/testing" "${DATA_ROOT}/testing"
        cp -R ../data/kitti/ImageSets/. "${IMAGESETS_ROOT}/"

        python triada/nap/nap_frames_select_merged.py \
            --train-detections "${SOURCE_DETECTIONS}" \
            --test-detections "${TARGET_DETECTIONS}" \
            --dataset-type kitti \
            --budgets "${BUDGET}" \
            --device "${NAP_DEVICE}" \
            --topk "${TOPK}" \
            --subset-name "${SUBSET_NAME}" \
            --emit-kitti-imagesets \
            --imagesets-root "${IMAGESETS_ROOT}" \
            --output "${OUT_ROOT}"

        if [[ "${OBJECT_CLASS}" == "pedestrian" ]]; then
            python triada/prepare_pedestrian_kitti_subset.py \
                --data-root "${DATA_ROOT}"
        else
            python -m pcdet.datasets.kitti.kitti_dataset \
                create_kitti_infos \
                "${DATA_CFG}"
        fi
    else
        DATA_CFG="cfgs/triada/data/target_nuscenes_budget_${BUDGET}.yaml"
        mkdir -p "${NUSCENES_SUBSET_PKLS_PATH}"

        python triada/nap/nap_frames_select_merged.py \
            --train-detections "${SOURCE_DETECTIONS}" \
            --test-detections "${TARGET_DETECTIONS}" \
            --dataset-type nuscenes \
            --budgets "${BUDGET}" \
            --device "${NAP_DEVICE}" \
            --topk "${TOPK}" \
            --subset-name "${SUBSET_NAME}" \
            --output "${OUT_ROOT}"

        python -m pcdet.datasets.nuscenes.nuscenes_dataset \
            --func create_nuscenes_infos \
            --cfg_file "${DATA_CFG}" \
            --version v1.0-trainval \
            --subset_json "${SUBSET_JSON}" \
            --data_path "${NUSCENES_DATA_PATH}" \
            --pkls_path "${NUSCENES_SUBSET_PKLS_PATH}"
    fi
done

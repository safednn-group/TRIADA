#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage:
  tools/triada/extract_nap_activations.sh EXPERIMENT source|target CHECKPOINT [test.py args...]

Experiments:
  waymo16_to_nuscenes
  waymo16_to_kitti
  nuscenes_to_kitti

Examples:
  tools/triada/extract_nap_activations.sh waymo16_to_nuscenes source checkpoints/pretrain_waymo16.pth
  tools/triada/extract_nap_activations.sh waymo16_to_nuscenes target checkpoints/pretrain_waymo16.pth --data_path data/nuscenes/v1.0-trainval --pkls_path data/nuscenes/v1.0-trainval

Environment:
  TRIADA_OBJECT_CLASS  car (default) or pedestrian
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
SPLIT_KIND=$2
CHECKPOINT=$3
shift 3

CALLER_CWD=$(pwd)
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
TOOLS_DIR="${REPO_ROOT}/tools"
OBJECT_CLASS=${TRIADA_OBJECT_CLASS:-car}

case "${OBJECT_CLASS}" in
    car|pedestrian)
        CFG_ROOT="cfgs/triada/${OBJECT_CLASS}"
        ;;
    *)
        echo "Invalid TRIADA_OBJECT_CLASS ${OBJECT_CLASS}; expected car or pedestrian" >&2
        exit 2
        ;;
esac

if [[ "${CHECKPOINT}" != /* ]]; then
    CHECKPOINT="${CALLER_CWD}/${CHECKPOINT}"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    if [[ ( "$1" == "--data_path" || "$1" == "--pkls_path" ) && $# -ge 2 ]]; then
        DATA_ARGUMENT=$2
        if [[ "${DATA_ARGUMENT}" != /* ]]; then
            DATA_ARGUMENT="${CALLER_CWD}/${DATA_ARGUMENT}"
        fi
        EXTRA_ARGS+=("$1" "${DATA_ARGUMENT}")
        shift 2
    else
        EXTRA_ARGS+=("$1")
        shift
    fi
done

case "${SPLIT_KIND}:${EXPERIMENT}" in
    source:waymo16_to_nuscenes|source:waymo16_to_kitti)
        CFG_FILE="${CFG_ROOT}/nap_activation/source_waymo16_ground_truth.yaml"
        ;;
    source:nuscenes_to_kitti)
        CFG_FILE="${CFG_ROOT}/nap_activation/source_nuscenes_ground_truth.yaml"
        ;;
    target:waymo16_to_nuscenes)
        CFG_FILE="${CFG_ROOT}/nap_activation/target_nuscenes_proposals_for_waymo16.yaml"
        ;;
    target:waymo16_to_kitti)
        CFG_FILE="${CFG_ROOT}/nap_activation/target_kitti_proposals.yaml"
        ;;
    target:nuscenes_to_kitti)
        CFG_FILE="${CFG_ROOT}/nap_activation/target_kitti_proposals.yaml"
        ;;
    *)
        usage
        exit 2
        ;;
esac

cd "${TOOLS_DIR}"
export PYTHONPATH="${REPO_ROOT}:${TOOLS_DIR}:${PYTHONPATH:-}"

python test.py \
    --cfg_file "${CFG_FILE}" \
    --ckpt "${CHECKPOINT}" \
    --skip_metric \
    "${EXTRA_ARGS[@]}"

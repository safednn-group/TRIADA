import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import entropy
from tqdm import tqdm

try:
    from .nap_monitor import DistanceMonitor, prepare_patterns
except ImportError:
    from nap_monitor import DistanceMonitor, prepare_patterns


# ---------------------------------------------------------
# utilities
# ---------------------------------------------------------


def count_labels(labels):
    if len(labels) == 0:
        raise ValueError("Source detections contain no activations")

    count_class = {}
    for c in range(int(labels.max() + 1)):
        count = (labels == c).sum()
        count_class[c] = int(count.item() if torch.is_tensor(count) else count)
    return count_class


def safe_stack(data, key):
    if not data:
        raise ValueError("Detection dump is empty")

    tensors = []

    for index, item in enumerate(data):
        value = item.get(key)
        if value is None:
            raise ValueError(f"Detection {index} has no '{key}' value")
        tensors.append(torch.as_tensor(value))

    if not tensors:
        raise ValueError(f"Detection dump contains no '{key}' values")
    return torch.cat(tensors)


# ---------------------------------------------------------
# monitor
# ---------------------------------------------------------


def build_monitor(monitor_cfg, detections):
    activations = safe_stack(detections, "activations")
    labels3d = np.zeros(activations.shape[0], dtype=int)

    monitor = DistanceMonitor(**monitor_cfg)

    monitor.set_class_patterns_count(count_labels(labels3d))
    monitor.add_neuron_pattern(activations, labels3d)
    monitor.cut_duplicates()

    return monitor


def compute_nap_distances(detections, monitor):
    for det in tqdm(detections):
        activations = torch.as_tensor(
            det["activations"],
            device=monitor.device,
            dtype=torch.float32,
        )

        labels3d = np.zeros(activations.shape[0], dtype=int)

        if activations.shape[0] > 0:
            distances = monitor.compute_distance(activations, labels3d)
        else:
            distances = np.zeros((0, len(monitor.layers_shapes)), dtype=int)

        det["nap_distance"] = distances

    return detections


# ---------------------------------------------------------
# feature extraction
# ---------------------------------------------------------


def compute_entropy(values):
    if len(values) == 0:
        return 0.0

    _, counts = np.unique(values, return_counts=True)
    probs = counts / counts.sum()

    return entropy(probs)


def build_frame_features(
    detections,
    dataset_type,
    device,
):
    rows = []

    for detection_index, det in enumerate(tqdm(detections)):
        num_predictions = len(det["boxes_lidar"])
        distances = np.asarray(det["nap_distance"])
        activations = torch.as_tensor(
            det["activations"],
            device=device,
            dtype=torch.float32,
        )
        if len(activations) != num_predictions or len(distances) != num_predictions:
            raise ValueError(
                f"Detection {detection_index} has inconsistent prediction, "
                "activation, and NAP-distance counts"
            )

        if dataset_type == "nuscenes":
            sample_token = det["metadata"]["token"]
        else:
            sample_token = str(det["frame_id"])

        if distances.shape[0] > 0:
            distance_entropy = compute_entropy(distances[:, 0])
        else:
            assert num_predictions == 0
            distance_entropy = 0.0

        if activations.shape[0] > 0:
            activations_256 = activations[:, :256]
            patterns = prepare_patterns(
                activations_256,
                [256],
                quantile=0.5,
                binarize=True,
            )
        else:
            patterns = torch.empty(
                (0, 256),
                device=device,
                dtype=torch.float32,
            )

        rows.append(
            {
                "sample_token": sample_token,
                "H_distance_l1": distance_entropy,
                "patterns": patterns,
            }
        )

    return rows


# ---------------------------------------------------------
# scoring
# ---------------------------------------------------------


def normalize(x):
    x = np.asarray(x)

    if x.max() == x.min():
        return np.ones_like(x)

    return (x - x.min()) / (x.max() - x.min())


def compute_scores(entropy_scores, inter_scores):
    entropy_scores = normalize(entropy_scores)
    inter_scores = normalize(inter_scores)

    return entropy_scores * inter_scores


# ---------------------------------------------------------
# selection
# ---------------------------------------------------------


def compute_interframe_distance(selected_patterns, candidate_patterns):
    box_distances = []

    for box_pattern in candidate_patterns:

        dist = torch.cdist(selected_patterns, box_pattern.unsqueeze(0), p=0).mean()

        box_distances.append(dist.item())

    return np.mean(box_distances)


def select_diverse_frames(rows, budget, topk=50):
    if budget <= 0:
        raise ValueError("Budget must be positive")
    if topk <= 0:
        raise ValueError("topk must be positive")
    # Empty frames cannot provide an inter-frame distance.
    eligible_rows = [row for row in rows if len(row["patterns"]) > 0]
    if budget > len(eligible_rows):
        raise ValueError(
            f"Budget {budget} exceeds the {len(eligible_rows)} "
            "target frames with retained proposals"
        )

    remaining = sorted(
        eligible_rows,
        key=lambda x: x["H_distance_l1"],
        reverse=True,
    )

    selected = []
    selected_patterns = []

    for selection_index in range(budget):
        candidates = remaining[:topk]

        if selection_index == 0:
            best_index = 0
        else:
            reference_patterns = torch.cat(selected_patterns)
            entropy_scores = []
            inter_scores = []

            for candidate in candidates:
                inter_scores.append(
                    compute_interframe_distance(
                        reference_patterns,
                        candidate["patterns"],
                    )
                )
                entropy_scores.append(candidate["H_distance_l1"])

            scores = compute_scores(entropy_scores, inter_scores)
            best_index = int(np.argmax(scores))

        best = remaining.pop(best_index)
        selected.append(best)
        selected_patterns.append(best["patterns"])

    return selected


# ---------------------------------------------------------
# subset helpers
# ---------------------------------------------------------


def save_kitti_frames_txt(tokens, out_txt):
    out_txt.parent.mkdir(parents=True, exist_ok=True)

    with out_txt.open("w") as f:
        for token in tokens:
            f.write(f"{token}\n")


def save_selection_json(selected_tokens, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(selected_tokens, f, indent=2)


# ---------------------------------------------------------
# main
# ---------------------------------------------------------


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[2]
    parser = argparse.ArgumentParser(
        description="Select diverse target frames with NAP."
    )
    parser.add_argument(
        "--train-detections", type=Path, required=True, help="source result.pkl"
    )
    parser.add_argument(
        "--test-detections", type=Path, required=True, help="target result.pkl"
    )
    parser.add_argument("--dataset-type", choices=("kitti", "nuscenes"), required=True)
    parser.add_argument("--budgets", nargs="+", type=int, required=True)
    parser.add_argument(
        "--device", default="cuda", help="PyTorch device used for NAP distances"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--subset-name",
        required=True,
        help="output stem for the selected subset",
    )
    parser.add_argument("--emit-kitti-imagesets", action="store_true")
    parser.add_argument(
        "--imagesets-root",
        type=Path,
        default=repo_root / "data" / "kitti" / "ImageSets",
    )
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error(f"CUDA is not available for --device {args.device!r}")
    if args.emit_kitti_imagesets and len(args.budgets) != 1:
        parser.error("--emit-kitti-imagesets requires exactly one budget")

    args.output.mkdir(parents=True, exist_ok=True)
    with args.train_detections.open("rb") as f:
        train_detections = pickle.load(f)
    with args.test_detections.open("rb") as f:
        test_detections = pickle.load(f)

    if not test_detections:
        raise ValueError("Target detection dump is empty")

    monitor_cfg = {
        "class_count": 1,
        "device": args.device,
        "layers_shapes": [256, 256, 256, 256],
        "quantile": 0.5,
        "p_norm": 0,
        "mean": False,
    }
    monitor = build_monitor(monitor_cfg, train_detections)
    monitor.to(args.device)

    print("Computing NAP distances...")
    test_detections = compute_nap_distances(test_detections, monitor)

    rows = build_frame_features(
        test_detections,
        args.dataset_type,
        args.device,
    )

    results = {}
    for budget in args.budgets:
        print(f"Selecting budget={budget}")
        results[budget] = select_diverse_frames(
            rows,
            budget=budget,
            topk=args.topk,
        )

    all_selected_tokens = {}
    for budget, selected in results.items():
        tokens = [item["sample_token"] for item in selected]
        all_selected_tokens[str(budget)] = tokens

        output_name = (
            f"{args.subset_name}.json"
            if len(args.budgets) == 1
            else f"{args.subset_name}_budget_{budget}.json"
        )
        budget_json = args.output / output_name
        save_selection_json(tokens, budget_json)

        if args.emit_kitti_imagesets:
            save_kitti_frames_txt(tokens, args.imagesets_root / "train.txt")

        print(f"[saved] {budget_json}")

    if len(args.budgets) > 1:
        summary_json = args.output / f"{args.subset_name}_all_budgets.json"
        with summary_json.open("w") as f:
            json.dump(all_selected_tokens, f, indent=2)

    for k, v in all_selected_tokens.items():
        print(f"budget={k} -> {len(v)} frames")


if __name__ == "__main__":
    main()

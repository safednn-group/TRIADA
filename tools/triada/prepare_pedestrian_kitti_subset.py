#!/usr/bin/env python3
"""Build info files for an isolated Pedestrian KITTI NAP subset."""

import argparse
from pathlib import Path

import yaml
from easydict import EasyDict

from pcdet.datasets.kitti.kitti_dataset import create_kitti_infos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    required = (data_root / "ImageSets" / "train.txt", data_root / "training")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error("missing prepared KITTI subset paths: " + ", ".join(missing))

    # The info schema is class-agnostic; the Pedestrian fine-tuning config
    # filters annotations to its one-class vocabulary when it loads these files.
    cfg_path = (
        Path(__file__).resolve().parents[1]
        / "cfgs/triada/data/target_kitti_waymo16_budget_10.yaml"
    )
    dataset_cfg = EasyDict(yaml.safe_load(cfg_path.read_text()))
    dataset_cfg.DATA_PATH = str(data_root)
    create_kitti_infos(
        dataset_cfg=dataset_cfg,
        class_names=["Car", "Pedestrian", "Cyclist"],
        data_path=data_root,
        save_path=data_root,
    )


if __name__ == "__main__":
    main()

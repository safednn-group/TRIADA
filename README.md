# TRIADA

TRIADA is a PV-RCNN-based domain-adaptation workflow for Waymo→nuScenes,
Waymo→KITTI, and nuScenes→KITTI.

## Quick start

Install the environment with [docs/INSTALL.md](docs/INSTALL.md), prepare the
datasets and source checkpoints, then run:

```bash
tools/triada/run_grid.sh \
  /path/to/waymo2kitti_checkpoint.pth \
  /path/to/waymo2nuscenes_checkpoint.pth \
  /path/to/nuscenes2kitti_checkpoint.pth
```

For a single run, use:

```bash
tools/triada/run_best_constant_finetune.sh SHIFT BUDGET CHECKPOINT
```

Use `TRIADA_OBJECT_CLASS=pedestrian` for Pedestrian runs. Configs are grouped
under `tools/cfgs/triada/car/` and `tools/cfgs/triada/pedestrian/`; shared
target-subset configs are in `tools/cfgs/triada/data/`.

See [docs/REPRODUCING.md](docs/REPRODUCING.md) for data preparation and
commands. Datasets and checkpoints are not included.

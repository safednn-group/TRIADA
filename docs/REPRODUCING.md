# Reproducing TRIADA

Install the environment first: [INSTALL.md](INSTALL.md).

## Data and checkpoints

Prepare the standard KITTI and nuScenes OpenPCDet data/info layouts under
`data/`, and the LiDAR-Distillation Waymo `modes/16^` layout. The supplied
configs expect:

- KITTI: `data/kitti/training/`, `data/kitti/testing/`, the versioned
  `data/kitti/ImageSets/` splits, and `kitti_infos_train.pkl` /
  `kitti_infos_val.pkl` in `data/kitti/`.
- nuScenes: samples, sweeps, and the `v1.0-trainval/` metadata directory
  below `data/nuscenes/v1.0-trainval/`, plus
  `nuscenes_infos_1sweeps_train.pkl` / `nuscenes_infos_1sweeps_val.pkl` in
  that same root. All supplied configs use one sweep. For a different
  layout, set `TRIADA_NUSCENES_DATA_PATH` to the directory containing
  samples, sweeps, and version metadata, and `TRIADA_NUSCENES_PKLS_PATH`
  to the directory containing the info files. Pass these paths explicitly
  to `train.py` when pretraining.
- Waymo: downsampled point clouds at
  `data/waymo/modes/16^/<sequence>/<frame>.npy` and sequence metadata at
  `data/waymo/waymo_processed_data_v0_5_0/<sequence>/<sequence>.pkl`, with
  the versioned `data/waymo/ImageSets/` splits. The `modes/16^` point clouds
  alone are insufficient; the dataset loader reads the separate metadata.

Datasets, generated info files, and source checkpoints are not included.

Source-pretraining configs are in:

- `tools/cfgs/triada/car/pretrain/`
- `tools/cfgs/triada/pedestrian/pretrain/`

For example, train the Car source models with:

```bash
cd tools

python train.py \
  --cfg_file cfgs/triada/car/pretrain/source_waymo16_lidar_aligned_pvrcnn.yaml \
  --no_eval

python train.py \
  --cfg_file cfgs/triada/car/pretrain/source_nuscenes_pvrcnn.yaml \
  --data_path ../data/nuscenes/v1.0-trainval \
  --pkls_path ../data/nuscenes/v1.0-trainval \
  --no_eval

cd ..
```

## Run Car

```bash
TRIADA_FULL_GRID_TAG=run_1 tools/triada/run_grid.sh \
  /path/to/waymo2kitti_checkpoint.pth \
  /path/to/waymo2nuscenes_checkpoint.pth \
  /path/to/nuscenes2kitti_checkpoint.pth
```

To run one shift/budget pair instead:

```bash
tools/triada/run_best_constant_finetune.sh \
  waymo2kitti 10 /path/to/waymo_checkpoint.pth
```

## Run Pedestrian

Train the source models with the Pedestrian configs above, then run:

```bash
TRIADA_OBJECT_CLASS=pedestrian \
TRIADA_FULL_GRID_TAG=run_1 \
  tools/triada/run_grid.sh \
    /path/to/pedestrian_waymo2kitti_checkpoint.pth \
    /path/to/pedestrian_waymo2nuscenes_checkpoint.pth \
    /path/to/pedestrian_nuscenes2kitti_checkpoint.pth
```

Car and Pedestrian configs are sibling trees under `tools/cfgs/triada/`.
Shared target-subset configs are in `tools/cfgs/triada/data/`.

Use a distinct `TRIADA_FULL_GRID_TAG` for each independent run. Outputs are
written below `output/triada/`; selected subsets are written to
`tools/triada_subsets/`.

The full grid runs nine Car cells or six Pedestrian cells. Activations are
cached separately for each dataset shift under
`output/triada/<class>/nap_activation/<config>/<grid_tag>_<shift>/`.
Reusing a tag reuses existing activation dumps; use a new tag when changing
checkpoints or data, or set `TRIADA_FORCE_EXTRACT=1` to regenerate them.
For single runs, set `TRIADA_RUN_TAG` to distinguish output directories.

Run independent Car experiments sequentially: their generated subset JSON
and dataset-info paths are shared even when output tags differ.
Pedestrian subsets use `TRIADA_SUBSET_TAG` (derived from the run tag by
default). Keep the selected JSON files and run settings alongside results
if you need to repeat an exact selection. `TRIADA_NAP_DEVICE=cpu` runs NAP
selection on the CPU; detector inference and training still require CUDA.

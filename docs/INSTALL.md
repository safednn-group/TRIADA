# Installation

The validated environment is Linux x86-64 with an NVIDIA GPU, Python 3.10,
PyTorch 2.5.1, CUDA 12.4, and spconv 2.3.8. Conda, `g++`, and an NVIDIA driver
supporting CUDA 12.4 are required.

From the repository root:

```bash
conda env create --file environment.yml
conda activate triada
python -m pip install --verbose --no-build-isolation --editable .
```

This builds the OpenPCDet CUDA extensions used by PV-RCNN. Verify the checkout,
GPU, and one compiled extension:

```bash
python -c "import pcdet, torch; from pcdet.ops.iou3d_nms import iou3d_nms_cuda; print(pcdet.__file__); assert torch.cuda.is_available()"
```

Keep Waymo TFRecord conversion and the official Waymo TensorFlow metric in a
separate environment; the workflow consumes already processed Waymo
data.

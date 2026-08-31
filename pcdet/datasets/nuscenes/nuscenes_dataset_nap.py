from ..nap_dataset_mixin import NAPDatasetMixin
from .nuscenes_dataset import NuScenesDataset


class NAPNuScenesDataset(NAPDatasetMixin, NuScenesDataset):
    pass

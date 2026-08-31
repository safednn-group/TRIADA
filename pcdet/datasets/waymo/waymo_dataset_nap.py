from ..nap_dataset_mixin import NAPDatasetMixin
from .waymo_dataset import WaymoDataset


class NAPWaymoDataset(NAPDatasetMixin, WaymoDataset):
    pass
